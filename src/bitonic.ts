function bitonicSortShader(itemsPerThread: number): string {
    return `
struct Data {
    values: array<f32>,
};

// Uniform buffer to store j and k
struct Uniforms {
    j: u32,
    k: u32,
};

@binding(0) @group(0) var<storage, read_write> data: Data;
@binding(1) @group(0) var<uniform> uniforms: Uniforms;

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
            let temp = data.values[i];
            data.values[i] = data.values[ixj];
            data.values[ixj] = temp;
        }
        if ((i & k) != 0 && data.values[i] < data.values[ixj]) {
            let temp = data.values[i];
            data.values[i] = data.values[ixj];
            data.values[ixj] = temp;
        }
    }
}
`;
}

function nextPowerOfTwo(x: number): number {
    return Math.pow(2, Math.ceil(Math.log2(x)));
}

class GpuContext {
    gpu: GPU;
    adapter: GPUAdapter;
    device: GPUDevice;

    constructor(gpu: GPU, adapter: GPUAdapter, device: GPUDevice) {
        this.gpu = gpu;
        this.adapter = adapter;
        this.device = device;
    }

    static async create(): Promise<GpuContext> {
        const gpu = navigator.gpu;
        if (!gpu) {
            return Promise.reject("WebGPU not supported on this browser! (navigator.gpu is null)");
        }

        const adapter = await gpu.requestAdapter();
        if (!adapter) {
            return Promise.reject("WebGPU not supported on this browser! (gpu.adapter is null)");
        }

        const device = await adapter.requestDevice();

        return new GpuContext(gpu, adapter, device);
    }
}

export class BitonicSorter {
    context: GpuContext;

    nElements: number;
    numThreads: number;

    pipeline: GPUComputePipeline;
    bindGroup: GPUBindGroup;

    dataBuffer: GPUBuffer;
    uniformsBuffer: GPUBuffer;

    constructor(context: GpuContext, nElements: number) {
        if (Math.log2(nElements) % 1 != 0) {
            throw new Error("nElements must be a power of 2");
        }

        this.context = context;
        this.nElements = nElements;
        this.dataBuffer = this.context.device.createBuffer({
            size: this.nElements * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            mappedAtCreation: false,
        });

        const uniformBufferSize = 8; // Size of two uint32 values
        this.uniformsBuffer = this.context.device.createBuffer({
            size: uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
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
                        buffer: this.dataBuffer
                    }
                },
                {
                    binding: 1,
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

    /* Sorts the values in the input buffer and returns the sorted values in a new buffer.
        Input size must be a power of two. */
    sort(values: GPUBuffer): GPUBuffer {
        if (values.size != this.dataBuffer.size) {
            throw new Error("Input buffer size does not match the size of the sorter");
        }

        // Copy the data to the GPU
        {
            const commandEncoder = this.context.device.createCommandEncoder();
            commandEncoder.clearBuffer(this.dataBuffer);
            commandEncoder.copyBufferToBuffer(values, 0, this.dataBuffer, 0, values.size);
            this.context.device.queue.submit([commandEncoder.finish()]);
        }

        // Sort
        for (let k = 2; k <= this.nElements; k <<= 1) {
            for (let j = k >> 1; j > 0; j = j >> 1) {
                const commandEncoder = this.context.device.createCommandEncoder();

                this.setUniforms(j, k);
                const passEncoder = commandEncoder.beginComputePass();
                passEncoder.setPipeline(this.pipeline);
                passEncoder.setBindGroup(0, this.bindGroup);
                passEncoder.dispatchWorkgroups(this.numThreads);
                passEncoder.end();
                this.context.device.queue.submit([commandEncoder.finish()]);
            }
        }

        // Read back the data
        {
            const commandEncoder = this.context.device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(this.dataBuffer, 0, values, 0, values.size);
            this.context.device.queue.submit([commandEncoder.finish()]);
        }

        return values;
    }
}

export function testBitonic() {
    // Usage example
    const values: Float32Array = new Float32Array(1 << 10);
    for (let i = 0; i < values.length; i++) {
        values[i] = Math.random();
    }
    
    // reference CPU sort
    const sorted = values.slice().sort((a, b) => a - b);
    console.log(sorted);

    // GPU sort
    GpuContext.create().then(context => {
        const sorter = new BitonicSorter(context, values.length);
        const valuesBuffer = context.device.createBuffer({
            size: values.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(valuesBuffer.getMappedRange()).set(values);
        valuesBuffer.unmap();

        const sortedBuffer = sorter.sort(valuesBuffer);

        const readBuffer = context.device.createBuffer({
            size: values.byteLength,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            mappedAtCreation: false,
        });
        const commandEncoder = context.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(sortedBuffer, 0, readBuffer, 0, values.byteLength);
        context.device.queue.submit([commandEncoder.finish()]);

        readBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const result = new Float32Array(readBuffer.getMappedRange());
            console.log(result);
            readBuffer.unmap();
        });
    });
}