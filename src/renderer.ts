// This file contains the main rendering code. Unlike the official implementation,
// instead of using compute shaders and iterating through (possibly) all gaussians,
// we instead use a vertex shader to turn each gaussian into a quad facing the camera
// and then use the fragment shader to paint the gaussian on the quad.
// If we draw the quads in order of depth, with well chosen blending settings we can
// get the same color accumulation rule as in the original paper.
// This approach is faster than the original implementation on webGPU but still substantially
// slow compared to the CUDA impl. The main bottleneck is the sorting of the quads by depth,
// which is done on the CPU but could presumably be replaced by a compute shader sort.

import { PackedGaussians } from './ply';
import { f32, Struct, vec3, mat4x4 } from './packing';
import { Camera, InteractiveCamera } from './camera';
import { getShaderCode } from './shaders';
import { Mat4, Vec3 } from 'wgpu-matrix';

const uniformLayout = new Struct([
    ['viewMatrix', new mat4x4(f32)],
    ['projMatrix', new mat4x4(f32)],
    ['cameraPosition', new vec3(f32)],
    ['tanHalfFovX', f32],
    ['tanHalfFovY', f32],
    ['focalX', f32],
    ['focalY', f32],
    ['scaleModifier', f32],
]);

function mat4toArrayOfArrays(m: Mat4): number[][] {
    return [
        [m[0], m[1], m[2], m[3]],
        [m[4], m[5], m[6], m[7]],
        [m[8], m[9], m[10], m[11]],
        [m[12], m[13], m[14], m[15]],
    ];
}

export class Renderer {
    canvas: HTMLCanvasElement;
    interactiveCamera: InteractiveCamera;

    adapter: GPUAdapter;
    device: GPUDevice;
    contextGpu: GPUCanvasContext;

    uniformBuffer: GPUBuffer;
    pointDataBuffer: GPUBuffer;
    drawIndexBuffer: GPUBuffer;
    drawIndexWriteBuffer: GPUBuffer;
    drawOrder: number[];
    pointPositions: Vec3[];

    pipeline: GPURenderPipeline;

    destroyCallback: (() => void) | null = null;

    // we need an async init function because we need to request certain async methods
    public static async init(canvas: HTMLCanvasElement, gaussians: PackedGaussians, interactiveCamera: InteractiveCamera): Promise<Renderer> {
        if (!canvas) {
            return Promise.reject("Canvas not found!");
        }

        if (!navigator.gpu) {
            return Promise.reject("WebGPU not supported on this browser! (navigator.gpu is null)");
        }
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            return Promise.reject("WebGPU is not supported on this browser! (WebGPU adapter not found)");
        }

        const byteLength = gaussians.gaussiansBuffer.byteLength;
        // for good measure, we request 1.5 times the amount of memory we need
        const device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: 1.5 * byteLength,
                maxBufferSize: 1.5 * byteLength,
            }
        });

        const contextGpu = canvas.getContext("webgpu");
        if (!contextGpu) {
            return Promise.reject("WebGPU context not found!");
        }

        return new Renderer(canvas, interactiveCamera, gaussians, adapter, device, contextGpu);
    }

    // destroy the renderer and return a promise that resolves when it's done (after the next frame)
    public async destroy(): Promise<void> {
        return new Promise((resolve, reject) => {
            this.destroyCallback = resolve;
        });
    }

    private constructor(
        canvas: HTMLCanvasElement,
        interactiveCamera: InteractiveCamera,
        gaussians: PackedGaussians,
        adapter: GPUAdapter,
        device: GPUDevice,
        contextGPU: GPUCanvasContext,
    ) {
        this.canvas = canvas;
        this.interactiveCamera = interactiveCamera;
        this.adapter = adapter;
        this.device = device;
        this.contextGpu = contextGPU;

        const presentationFormat = "rgba16float" as GPUTextureFormat;

        this.contextGpu.configure({
            device,
            format: presentationFormat,
            alphaMode: 'premultiplied' as GPUCanvasAlphaMode,
        });

        this.pointDataBuffer = device.createBuffer({
            size: gaussians.gaussianArrayLayout.size,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
        });
        new Uint8Array(this.pointDataBuffer.getMappedRange()).set(new Uint8Array(gaussians.gaussiansBuffer));
        this.pointDataBuffer.unmap();

        // Create a GPU buffer for the uniform data.
        this.uniformBuffer = device.createBuffer({
            size: uniformLayout.size,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const shaderCode = getShaderCode(canvas, gaussians.sphericalHarmonicsDegree, gaussians.nShCoeffs);
        const shaderModule = device.createShaderModule({ code: shaderCode });

        this.pipeline = device.createRenderPipeline({
            layout: "auto",
            vertex: {
                module: shaderModule,
                entryPoint: "vs_points",
            },
            fragment: {
                module: shaderModule,
                entryPoint: "fs_main",
                targets: [
                    {
                        format: presentationFormat,
                        // with one-minus-dst alpha we can set the src to src.alpha * src.color and
                        // we get that color_new = src.color * src.alpha + dst.color * (1 - src.alpha)
                        // which is the same as the accumulation rule in the paper
                        blend: {
                            color: {
                                srcFactor: "one-minus-dst-alpha" as GPUBlendFactor,
                                dstFactor: "one" as GPUBlendFactor,
                                operation: "add" as GPUBlendOperation,
                            },
                            alpha: {
                                srcFactor: "one-minus-dst-alpha" as GPUBlendFactor,
                                dstFactor: "one" as GPUBlendFactor,
                                operation: "add" as GPUBlendOperation,
                            },
                        }
                    },
                ],
            },
            primitive: {
                topology: "triangle-list",
                stripIndexFormat: undefined,
                cullMode: undefined,
            },
        });

        // point positions for sorting by depth
        this.pointPositions = gaussians.positionsArray;
        // sorting is faster on partially sorted lists so we keep the old indices around,
        // initialized to the identity permutation 
        this.drawOrder = Array.from(Array(this.pointPositions.length).keys());
        
        // create a buffer with the draw order and a copy buffer for it
        this.drawIndexBuffer = device.createBuffer({
            size: 6 * 4 * this.drawOrder.length,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: false,
        });

        this.drawIndexWriteBuffer = device.createBuffer({
            size: 6 * 4 * this.drawOrder.length,
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE,
            mappedAtCreation: false,
        });

        // start the animation loop
        requestAnimationFrame(() => this.animate());
    }

    private destroyImpl(): void {
        if (this.destroyCallback === null) {
            throw new Error("destroyImpl called without destroyCallback set!");
        }

        this.uniformBuffer.destroy();
        this.pointDataBuffer.destroy();
        this.drawIndexBuffer.destroy();
        this.drawIndexWriteBuffer.destroy();
        this.device.destroy();
        this.adapter = null as any;
        this.device = null as any;
        this.contextGpu = null as any;
        this.pipeline = null as any;
        this.destroyCallback();
    }

    async draw(nextFrameCallback: FrameRequestCallback): Promise<void> {
        const commandEncoder = this.device.createCommandEncoder();

        // this.drawOrder is in terms of quads, but we draw vertices
        // so we need to convert the draw order to a vertex draw order
        const triangleDrawOrder = [];
        for (let i = 0; i < this.drawOrder.length; i++) {
            const quadIndex = this.drawOrder[i];
            for (let j = 0; j < 6; j++) {
                triangleDrawOrder.push(6 * quadIndex + j);
            }
        }

        // copy the draw order to the draw index buffer
        // first map the write buffer
        await this.drawIndexWriteBuffer.mapAsync(GPUMapMode.WRITE);
        new Uint32Array(this.drawIndexWriteBuffer.getMappedRange()).set(triangleDrawOrder);
        this.drawIndexWriteBuffer.unmap();

        // copy the write buffer to the draw index buffer
        commandEncoder.copyBufferToBuffer(
            this.drawIndexWriteBuffer,
            0,
            this.drawIndexBuffer,
            0,
            6 * 4 * this.drawOrder.length,
        );

        const textureView = this.contextGpu.getCurrentTexture().createView();
        const renderPassDescriptor: GPURenderPassDescriptor = {
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                storeOp: "store" as GPUStoreOp,
                loadOp: "clear" as GPULoadOp,
            }],
        };

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [{
                binding: 0,
                resource: {
                    buffer: this.uniformBuffer,
                },
            }],
        }));
        passEncoder.setBindGroup(1, this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(1),
            entries: [{
                binding: 1,
                resource: {
                    buffer: this.pointDataBuffer,
                },
            }],
        }));

        passEncoder.setIndexBuffer(this.drawIndexBuffer, "uint32" as GPUIndexFormat)
        passEncoder.drawIndexed(this.drawOrder.length * 2 * 3, 1, 0, 0, 0);
        passEncoder.end();

        this.device.queue.submit([commandEncoder.finish()]);

        if (this.destroyCallback === null) {
            requestAnimationFrame(nextFrameCallback);
        } else {
            this.destroyImpl();
        }
    }

    animate() {
        if (!this.interactiveCamera.isDirty()) {
            requestAnimationFrame(() => this.animate());
            return;
        }
        const camera = this.interactiveCamera.getCamera();

        const position = camera.getPosition();

        const tanHalfFovX = 0.5 * this.canvas.width / camera.focalX;
        const tanHalfFovY = 0.5 * this.canvas.height / camera.focalY;

        let uniformsMatrixBuffer = new ArrayBuffer(this.uniformBuffer.size);
        let uniforms = {
            viewMatrix: mat4toArrayOfArrays(camera.viewMatrix),
            projMatrix: mat4toArrayOfArrays(camera.getProjMatrix()),
            cameraPosition: Array.from(position),
            tanHalfFovX: tanHalfFovX,
            tanHalfFovY: tanHalfFovY,
            focalX: camera.focalX,
            focalY: camera.focalY,
            scaleModifier: camera.scaleModifier,

        };
        uniformLayout.pack(0, uniforms, new DataView(uniformsMatrixBuffer));

        this.device.queue.writeBuffer(
            this.uniformBuffer,
            0,
            uniformsMatrixBuffer,
            0,
            uniformsMatrixBuffer.byteLength
        );

        const depthProjection = camera.dotZ();
        const depthsWithIndices: [number, number][] = [];
        for (let i = 0; i < this.pointPositions.length; i++) {
            const index = this.drawOrder[i];
            const position = this.pointPositions[index];
            const depth = depthProjection(position);
            depthsWithIndices.push([depth, index]);
        }
        this.drawOrder = depthsWithIndices.sort(([d1, i1], [d2, i2]) => (d1 - d2)).map(([d, i]) => i);

        this.draw(() => this.animate());
    }
}