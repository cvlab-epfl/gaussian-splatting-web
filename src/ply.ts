import { PackingType, StaticArray, Struct, vec3, vec4, f32 } from "./packing";
import { Vec3 } from "wgpu-matrix";

export function loadFileAsArrayBuffer(file: File): Promise<ArrayBuffer> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = (event) => {
            if (!event.target || !event.target.result) {
                reject('Failed to load file');
                return;
            }
            if (typeof event.target.result === 'string') {
                reject('Got a text file instead of a binary one');
                return;
            }
            resolve(event.target.result);
        };

        reader.onerror = (event) => {
            if (!event.target) {
                reject('Failed to load file');
                return;
            }
            reject(event.target.error);
        };

        reader.readAsArrayBuffer(file);
    });
}

export class PackedGaussians {
    numGaussians: number;
    sphericalHarmonicsDegree: number;

    gaussianLayout: PackingType;
    public gaussianArrayLayout: PackingType;

    positionsArray: Vec3[];
    gaussiansBuffer: ArrayBuffer;

    private static decodeHeader(plyArrayBuffer: ArrayBuffer): [number, Record<string, string>, DataView] {
        const decoder = new TextDecoder();
        let headerOffset = 0;
        let headerText = '';

        while (true) {
            const headerChunk = new Uint8Array(plyArrayBuffer, headerOffset, 50);
            headerText += decoder.decode(headerChunk);
            headerOffset += 50;

            if (headerText.includes('end_header')) {
                break;
            }
        }

        const headerLines = headerText.split('\n');

        let vertexCount = 0;
        let propertyTypes: Record<string, string> = {};

        for (let i = 0; i < headerLines.length; i++) {
            const line = headerLines[i].trim();
            if (line.startsWith('element vertex')) {
                const vertexCountMatch = line.match(/\d+/);
                if (vertexCountMatch) {
                    vertexCount = parseInt(vertexCountMatch[0]);
                }
            } else if (line.startsWith('property')) {
                const propertyMatch = line.match(/(\w+)\s+(\w+)\s+(\w+)/);
                if (propertyMatch) {
                    const propertyType = propertyMatch[2];
                    const propertyName = propertyMatch[3];
                    propertyTypes[propertyName] = propertyType;
                }
            } else if (line === 'end_header') {
                break;
            }
        }

        const vertexByteOffset = headerText.indexOf('end_header') + 'end_header'.length + 1;
        const vertexData = new DataView(plyArrayBuffer, vertexByteOffset);

        return [
            vertexCount,
            propertyTypes,
            vertexData,
        ];
    }

    private readRawVertex(offset: number, vertexData: DataView, propertyTypes: Record<string, string>): [number, Record<string, number>] {
        let rawVertex: Record<string, number> = {};

        for (const property in propertyTypes) {
            const propertyType = propertyTypes[property];
            if (propertyType === 'float') {
                rawVertex[property] = vertexData.getFloat32(offset, true);
                offset += Float32Array.BYTES_PER_ELEMENT;
            } else if (propertyType === 'uchar') {
                rawVertex[property] = vertexData.getUint8(offset) / 255.0;
                offset += Uint8Array.BYTES_PER_ELEMENT;
            }
        }

        return [offset, rawVertex];
    }

    public get nShCoeffs(): number {
        if (this.sphericalHarmonicsDegree === 0) {
            return 1;
        } else if (this.sphericalHarmonicsDegree === 1) {
            return 4;
        } else if (this.sphericalHarmonicsDegree === 2) {
            return 9;
        } else if (this.sphericalHarmonicsDegree === 3) {
            return 16;
        } else {
            throw new Error(`Unsupported SH degree: ${this.sphericalHarmonicsDegree}`);
        }
    }

    private arrangeVertex(rawVertex: Record<string, number>, shFeatureOrder: string[]): Record<string, any> {
        const shCoeffs = [];
        for (let i = 0; i < this.nShCoeffs; ++i) {
            const coeff = [];
            for (let j = 0; j < 3; ++j) {
                const coeffName = shFeatureOrder[i * 3 + j];
                coeff.push(rawVertex[coeffName]);
            }
            shCoeffs.push(coeff);
        }

        const arrangedVertex: Record<string, any> = {
            position: [rawVertex.x, rawVertex.y, rawVertex.z],
            logScale: [rawVertex.scale_0, rawVertex.scale_1, rawVertex.scale_2],
            rotQuat: [rawVertex.rot_0, rawVertex.rot_1, rawVertex.rot_2, rawVertex.rot_3],
            opacityLogit: rawVertex.opacity,
            shCoeffs: shCoeffs,
        };
        return arrangedVertex;
    }

    constructor(arrayBuffer: ArrayBuffer) {
        // decode the header
        const [vertexCount, propertyTypes, vertexData] = PackedGaussians.decodeHeader(arrayBuffer);
        this.numGaussians = vertexCount;

        // figure out the SH degree from the number of coefficients
        var nRestCoeffs = 0;
        for (const propertyName in propertyTypes) {
            if (propertyName.startsWith('f_rest_')) {
                nRestCoeffs += 1;
            }
        }
        const nCoeffsPerColor = nRestCoeffs / 3;
        this.sphericalHarmonicsDegree = Math.sqrt(nCoeffsPerColor + 1) - 1;
        console.log('Detected degree', this.sphericalHarmonicsDegree, 'with ', nCoeffsPerColor, 'coefficients per color');

        // figure out the order in which spherical harmonics should be read
        const shFeatureOrder = [];
        for (let rgb = 0; rgb < 3; ++rgb) {
            shFeatureOrder.push(`f_dc_${rgb}`);
        }
        for (let i = 0; i < nCoeffsPerColor; ++i) {
            for (let rgb = 0; rgb < 3; ++rgb) {
                shFeatureOrder.push(`f_rest_${rgb * nCoeffsPerColor + i}`);
            }
        }

        // define the layout of a single point
        this.gaussianLayout = new Struct([
            ['position', new vec3(f32)],
            ['logScale', new vec3(f32)],
            ['rotQuat', new vec4(f32)],
            ['opacityLogit', f32],
            ['shCoeffs', new StaticArray(new vec3(f32), this.nShCoeffs)],
        ]);
        // define the layout of the entire point cloud
        this.gaussianArrayLayout = new StaticArray(this.gaussianLayout, vertexCount);

        // pack the points
        this.positionsArray = [];
        this.gaussiansBuffer = new ArrayBuffer(this.gaussianArrayLayout.size);
        const gaussianWriteView = new DataView(this.gaussiansBuffer);

        var readOffset = 0;
        var gaussianWriteOffset = 0;
        for (let i = 0; i < vertexCount; i++) {
            const [newReadOffset, rawVertex] = this.readRawVertex(readOffset, vertexData, propertyTypes);
            readOffset = newReadOffset;
            gaussianWriteOffset = this.gaussianLayout.pack(
                gaussianWriteOffset,
                this.arrangeVertex(rawVertex, shFeatureOrder),
                gaussianWriteView,
            );

            this.positionsArray.push([rawVertex.x, rawVertex.y, rawVertex.z]);
        }
    }
}