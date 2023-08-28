import { GpuContext } from "./gpu_context";

function bitonicSortShader(itemsPerThread: number): string {
    return `
struct Data {
    values: array<f32>,
};

struct Indices {
    values: array<u32>,
};

// Uniform buffer to store j and k
struct Uniforms {
    j: u32,
    k: u32,
};

@binding(0) @group(0) var<storage, read_write> data: Data;
@binding(1) @group(0) var<storage, read_write> indices: Indices;
@binding(2) @group(0) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = uniforms.j;
    let k = uniforms.k;
    
    for (var i = global_id.x * ${itemsPerThread}; i < (global_id.x + 1) * ${itemsPerThread}; i++) {
        let ixj = i ^ j;

        if (ixj <= i) {
            continue;
        }

        if ((i & k) == 0 && data.values[i] > data.values[ixj]) {
            let tempV = data.values[i];
            data.values[i] = data.values[ixj];
            data.values[ixj] = tempV;

            let tempI = indices.values[i];
            indices.values[i] = indices.values[ixj];
            indices.values[ixj] = tempI;
        }
        if ((i & k) != 0 && data.values[i] < data.values[ixj]) {
            let tempV = data.values[i];
            data.values[i] = data.values[ixj];
            data.values[ixj] = tempV;

            let tempI = indices.values[i];
            indices.values[i] = indices.values[ixj];
            indices.values[ixj] = tempI;
        }
    }
}
`;
}

export class BitonicSorter {
    context: GpuContext;

    nElements: number;
    numThreads: number;

    pipeline: GPUComputePipeline;
    bindGroup: GPUBindGroup;

    valuesBuffer: GPUBuffer; // The buffer to sort
    indicesBuffer: GPUBuffer; // The indices of the values (to be sorted)
    initialIndexBuffer: GPUBuffer; // The initial indices of the values, to copy from at each call
    uniformsBuffer: GPUBuffer;

    constructor(context: GpuContext, nElements: number) {
        if (Math.log2(nElements) % 1 != 0) {
            throw new Error("nElements must be a power of 2");
        }

        this.context = context;
        this.nElements = nElements;
        this.valuesBuffer = this.context.device.createBuffer({
            size: this.nElements * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            mappedAtCreation: false,
            label: "bitonicSorter.valuesBuffer"
        });

        this.indicesBuffer = this.context.device.createBuffer({
            size: this.nElements * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            mappedAtCreation: false,
            label: "bitonicSorter.indicesBuffer"
        });

        this.initialIndexBuffer = this.context.device.createBuffer({
            size: this.nElements * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
            label: "bitonicSorter.initialIndexBuffer"
        });

        const initialIndices = new Uint32Array(this.initialIndexBuffer.getMappedRange());
        for (let i = 0; i < this.nElements; i++) {
            initialIndices[i] = i;
        }
        this.initialIndexBuffer.unmap();

        const uniformBufferSize = 8; // Size of two uint32 values
        this.uniformsBuffer = this.context.device.createBuffer({
            size: uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "bitonicSorter.uniformsBuffer"
        });

        this.numThreads = 1024;
        const itemsPerThread = Math.ceil(this.nElements / this.numThreads);
        this.pipeline = this.context.device.createComputePipeline({
            compute: {
                module: this.context.device.createShaderModule({
                    code: bitonicSortShader(itemsPerThread),
                }),
                entryPoint: 'main',
            },
            layout: 'auto',
        });

        this.bindGroup = this.context.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.valuesBuffer
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.indicesBuffer,
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.uniformsBuffer
                    }
                }
            ],
        });
    }

    private setUniforms(j: number, k: number): void {
        const uniformsArray = new Uint32Array([j, k]);
        this.context.device.queue.writeBuffer(this.uniformsBuffer, 0, uniformsArray.buffer);
    };

    argsort(values: GPUBuffer): GPUBuffer {
        if (values.size != this.valuesBuffer.size) {
            throw new Error("Input buffer size does not match the size of the sorter");
        }

        // Copy the data to the GPU
        {
            const commandEncoder = this.context.device.createCommandEncoder();
            // clear just in case
            commandEncoder.clearBuffer(this.valuesBuffer);
            // copy the values
            commandEncoder.copyBufferToBuffer(values, 0, this.valuesBuffer, 0, values.size);

            // write the initial indices
            commandEncoder.copyBufferToBuffer(this.initialIndexBuffer, 0, this.indicesBuffer, 0, this.initialIndexBuffer.size);

            this.context.device.queue.submit([commandEncoder.finish()]);
        }

        // Sort
        for (let k = 2; k <= this.nElements; k <<= 1) {
            for (let j = k >> 1; j > 0; j = j >> 1) {
                this.setUniforms(j, k);

                const commandEncoder = this.context.device.createCommandEncoder();
                const passEncoder = commandEncoder.beginComputePass();
                passEncoder.setPipeline(this.pipeline);
                passEncoder.setBindGroup(0, this.bindGroup);
                passEncoder.dispatchWorkgroups(this.numThreads);
                passEncoder.end();
                this.context.device.queue.submit([commandEncoder.finish()]);
            }
        }

        return this.indicesBuffer;
    }
}

export function testBitonic() {
    // Usage example
    const values: Float32Array = new Float32Array(1 << 10);
    for (let i = 0; i < values.length; i++) {
        values[i] = Math.random();
    }
    
    // reference CPU argsort
    const valuesWithIndices: [number, number][] = [];
    for (let i = 0; i < values.length; i++) {
        valuesWithIndices.push([values[i], i]);
    }
    valuesWithIndices.sort((a, b) => a[0] - b[0]);
    const cpuResult = new Uint32Array(values.length);
    for (let i = 0; i < values.length; i++) {
        cpuResult[i] = valuesWithIndices[i][1];
    }
    console.log(cpuResult);

    // GPU argsort
    GpuContext.create().then(context => {
        const sorter = new BitonicSorter(context, values.length);
        const valuesBuffer = context.device.createBuffer({
            size: values.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
            label: "testBitonic.valuesBuffer"
        });
        new Float32Array(valuesBuffer.getMappedRange()).set(values);
        valuesBuffer.unmap();

        const argSortBuffer = sorter.argsort(valuesBuffer);

        const readBuffer = context.device.createBuffer({
            size: values.byteLength,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            mappedAtCreation: false,
            label: "testBitonic.readBuffer"
        });
        const commandEncoder = context.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(argSortBuffer, 0, readBuffer, 0, values.byteLength);
        context.device.queue.submit([commandEncoder.finish()]);

        readBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const result = new Uint32Array(readBuffer.getMappedRange());
            console.log(result);
            readBuffer.unmap();
        });
    });
}