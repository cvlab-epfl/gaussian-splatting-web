// This file contains the code for packing and unpacking data into webGPU buffers.
// WebGPU buffers have rules about datatype layouts, padding etc, so it's much easier
// to define a code which automatically generates the correct packing methods.
// It is weakly typed and relies on runtime check, a proper solution would be with
// extensive generics, but that's outside my comfort zone.

// I wished to define it as 
// export type NestedData = number | NestedData[] | Record<string, NestedData>;
// but it complains about circular references.
export type NestedData = any;

function roundUp(n: number, multiple: number): number {
    return Math.ceil(n / multiple) * multiple;
}

class PackingError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'PackingError';
    }
}

export abstract class PackingType {
    public size: number;
    public alignment: number;

    constructor(size: number, alignment: number) {
        this.size = size;
        this.alignment = alignment;
    }

    abstract pack(offset: number, value: NestedData, view: DataView): number;
    abstract unpack(offset: number, view: DataView): [number, NestedData];
}

class i32Type extends PackingType {
    constructor() { super(4, 4); }
    pack(offset: number, value: number, view: DataView): number {
        if (typeof value !== 'number') {
            throw new PackingError(`Expected number, got ${value}`);
        }
        view.setInt32(offset, value, true);
        return offset + this.size;
    }

    unpack(offset: number, view: DataView): [number, number] {
        const content = view.getInt32(offset, true);
        return [offset + this.size, content];
    }
}

class u32Type extends PackingType {
    constructor() { super(4, 4); }
    pack(offset: number, value: number, view: DataView): number {
        if (typeof value !== 'number') {
            throw new PackingError(`Expected number, got ${value}`);
        }
        view.setUint32(offset, value, true);
        return offset + this.size;
    }

    unpack(offset: number, view: DataView): [number, number] {
        const content = view.getUint32(offset, true);
        return [offset + this.size, content];
    }

}

class f32Type extends PackingType {
    constructor() { super(4, 4); }
    pack(offset: number, value: number, view: DataView): number {
        if (typeof value !== 'number') {
            throw new PackingError(`Expected number, got ${value}`);
        }
        view.setFloat32(offset, value, true);
        return offset + this.size;
    }

    unpack(offset: number, view: DataView): [number, number] {
        const content = view.getFloat32(offset, true);
        return [offset + this.size, content];
    }
}

export const i32 = new i32Type();
export const u32 = new u32Type();
export const f32 = new f32Type();

class VectorType extends PackingType {
    public baseType: PackingType;
    public nValues: number;

    constructor(baseType: PackingType, nValues: number, alignment: number) {
        super(baseType.size * nValues, alignment);
        this.baseType = baseType;
        this.nValues = nValues;
    }

    pack(offset: number, values: number[], view: DataView) {
        if (!Array.isArray(values)) {
            throw new PackingError(`Expected array, got ${values}`);
        }

        if (values.length !== this.nValues) {
            throw new PackingError(`Expected ${this.nValues} values, got ${values.length}`);
        }

        while (offset % this.alignment !== 0) {
            offset++;
        }

        for (let i = 0; i < values.length; i++) {
            try {
                offset = this.baseType.pack(offset, values[i], view);
            } catch (e) {
                if (e instanceof PackingError) {
                    throw new PackingError(`Error packing value ${i}: ${e.message}`);
                } else {
                    throw e;
                }
            };
        }
        return offset;
    }

    unpack(offset: number, view: DataView): [number, number[]] {
        const values: number[] = [];

        while (offset % this.alignment !== 0) {
            offset++;
        }

        for (let i = 0; i < this.nValues; i++) {
            let [newOffset, value] = this.baseType.unpack(offset, view);
            offset = newOffset;
            values.push(value);
        }
        return [offset, values];
    }
}

export class vec2 extends VectorType {
    constructor(baseType: PackingType) { super(baseType, 2, 8); }
}

export class vec3 extends VectorType {
    constructor(baseType: PackingType) { super(baseType, 3, 16); }
}

export class vec4 extends VectorType {
    constructor(baseType: PackingType) { super(baseType, 4, 16); }
}

export class Struct extends PackingType {
    public members: [string, PackingType][];

    constructor(members: [string, PackingType][]) {
        const alignment = Math.max(...members.map(([_name, type]) => type.alignment));

        let offset = 0;
        for (const [_, type] of members) {
            while (offset % type.alignment !== 0) {
                offset++;
            }

            offset += type.size;
        }

        // SizeOf(S) = roundUp(AlignOf(S), justPastLastMember)
        // where justPastLastMember = OffsetOfMember(S,N) + SizeOfMember(S,N)
        const size = roundUp(offset, alignment);
        super(size, alignment);
        this.members = members;
    }

    pack(offset: number, values: Record<string, NestedData>, view: DataView) {
        const expectedKeys = this.members.map(([name, _type]) => name);
        const actualKeys = Object.keys(values);

        if (expectedKeys.length !== actualKeys.length) {
            throw new PackingError(`Expected values for ${expectedKeys}, got ${actualKeys}`);
        }
        if (!expectedKeys.every((key) => actualKeys.includes(key))) {
            throw new PackingError(`Expected values for ${expectedKeys}, got ${actualKeys}`);
        }

        const startingOffset = offset;

        while (offset % this.alignment !== 0) {
            offset++;
        }

        for (const [name, type] of this.members) {
            const value = values[name as keyof typeof values];
            try {
                offset = type.pack(offset, value, view);
            } catch (e) {
                // error packing the thing inside
                if (e instanceof PackingError) {
                    throw new PackingError(`Error packing value ${name}: ${e.message}`);
                } else {
                    throw e;
                }
            }
        }

        offset += this.size - (offset - startingOffset);

        return offset;
    }

    unpack(offset: number, view: DataView): [number, Record<string, NestedData>] {
        const values: Record<string, NestedData> = {};

        const startingOffset = offset;

        while (offset % this.alignment !== 0) {
            offset++;
        }

        for (const [name, type] of this.members) {
            let [newOffset, value] = type.unpack(offset, view);
            offset = newOffset;
            values[name] = value;
        }

        offset += this.size - (offset - startingOffset);

        return [offset, values];
    }
}

export class StaticArray extends PackingType {
    public type: PackingType;
    public nElements: number;
    public stride: number;

    constructor(type: PackingType, nElements: number) {
        const alignment = type.alignment;
        const size = nElements * roundUp(type.size, type.alignment);
        super(size, alignment);
        this.type = type
        this.nElements = nElements;
        this.stride = roundUp(type.size, type.alignment);
    }

    pack(offset: number, values: NestedData[], view: DataView) {
        if (!Array.isArray(values)) {
            throw new PackingError(`Expected array, got ${values}`);
        }

        if (values.length !== this.nElements) {
            throw new PackingError(`Expected ${this.nElements} values, got ${values.length}`);
        }

        while (offset % this.alignment !== 0) {
            offset++;
        }

        for (let i = 0; i < values.length; i++) {
            try {
                offset = this.type.pack(offset, values[i], view);
            } catch (e) {
                if (e instanceof PackingError) {
                    throw new PackingError(`Error packing value ${i}: ${e.message}`);
                } else {
                    throw e;
                }
            }
            offset += this.stride - this.type.size;
        }

        return offset;
    }

    unpack(offset: number, view: DataView): [number, NestedData[]] {
        const values: NestedData[] = [];

        while (offset % this.alignment !== 0) {
            offset++;
        }

        for (let i = 0; i < this.nElements; i++) {
            let [newOffset, value] = this.type.unpack(offset, view);
            offset = newOffset;
            values.push(value);
            offset += this.stride - this.type.size;
        }
        return [offset, values];
    }
}

class MatrixType extends PackingType {
    public baseType: PackingType;
    public nRows: number;
    public nColumns: number;

    constructor(baseType: PackingType, nRows: number, nColumns: number) {
        var vecType: VectorType;
        if (nRows === 2) {
            vecType = new vec2(baseType);
        } else if (nRows === 3) {
            vecType = new vec3(baseType);
        } else if (nRows === 4) {
            vecType = new vec4(baseType);
        } else {
            throw new Error(`Invalid number of rows: ${nRows}`);
        }
        const arrayType = new StaticArray(vecType, nColumns);

        super(arrayType.size, vecType.alignment);
        this.baseType = baseType;
        this.nRows = nRows;
        this.nColumns = nColumns;
    }

    pack(offset: number, values: number[][], view: DataView): number {
        if (!Array.isArray(values)) {
            throw new PackingError(`Expected array, got ${values}`);
        }

        if (values.length !== this.nColumns) {
            throw new PackingError(`Expected ${this.nColumns} columns, got ${values.length}`);
        }

        while (offset % this.alignment !== 0) {
            offset++;
        }

        const startOffset = offset;

        for (let i = 0; i < values.length; i++) {
            if (!Array.isArray(values[i])) {
                throw new PackingError(`Expected array, got ${values[i]}`);
            }

            for (let j = 0; j < values[i].length; j++) {
                try {
                    offset = this.baseType.pack(offset, values[i][j], view);
                } catch (e) {
                    if (e instanceof PackingError) {
                        throw new PackingError(`Error packing value ${i},${j}: ${e.message}`);
                    } else {
                        throw e;
                    }
                }
            }
        }

        offset = startOffset + this.size;

        return offset;
    }

    unpack(offset: number, view: DataView): [number, number[][]] {
        while (offset % this.alignment !== 0) {
            offset++;
        }

        const startOffset = offset;

        const outerValues: number[][] = [];
        for (let i = 0; i < this.nColumns; i++) {
            const innerValues: number[] = [];
            for (let j = 0; j < this.nRows; j++) {
                let [newOffset, value] = this.baseType.unpack(offset, view);
                offset = newOffset;
                innerValues.push(value);
            }
            outerValues.push(innerValues);
        }

        offset += this.size - (offset - startOffset);

        return [offset, outerValues];
    }
}

export class mat4x4 extends MatrixType {
    constructor(baseType: PackingType) { super(baseType, 4, 4); }
}