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

async function bitonicSortWebGPU(values: Float32Array): Promise<Float32Array> {
    // Initialization
    const gpu = navigator.gpu;
    if (!gpu) {
        return Promise.reject("WebGPU not supported on this browser! (navigator.gpu is null)");
    }

    const adapter = await gpu.requestAdapter();
    if (!adapter) {
        return Promise.reject("WebGPU not supported on this browser! (gpu.adapter is null)");
    }
    const device = await adapter.requestDevice();

    const paddedLength = nextPowerOfTwo(values.length);
    const paddedValues = new Float32Array(paddedLength);
    paddedValues.set(values);
    paddedValues.fill(Infinity, values.length);

    const dataBuffer = device.createBuffer({
        size: paddedValues.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });

    new Float32Array(dataBuffer.getMappedRange()).set(paddedValues);
    dataBuffer.unmap();

    const uniformBufferSize = 8; // Size of two uint32 values
    const uniformsBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const numThreads = 1024;
    const itemsPerThread = Math.ceil(paddedLength / numThreads);
    const pipeline = device.createComputePipeline({
        compute: {
            module: device.createShaderModule({
                code: bitonicSortShader(itemsPerThread),
            }),
            entryPoint: 'main',
        },
        layout: 'auto',
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: dataBuffer
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: uniformsBuffer
                }
            }
        ],
    });

    const setUniforms = (j: number, k: number) => {
        const uniformsArray = new Uint32Array([j, k]);
        device.queue.writeBuffer(uniformsBuffer, 0, uniformsArray.buffer);
    };

    for (let k = 2; k <= paddedLength; k <<= 1) {
       for (let j = k >> 1; j > 0; j = j >> 1) {
           const commandEncoder = device.createCommandEncoder();

           setUniforms(j, k);
           const passEncoder = commandEncoder.beginComputePass();
           passEncoder.setPipeline(pipeline);
           passEncoder.setBindGroup(0, bindGroup);
           passEncoder.dispatchWorkgroups(numThreads);
           passEncoder.end();
           device.queue.submit([commandEncoder.finish()]);
       }
    }

    // Read back the data
    const readBuffer = device.createBuffer({
        size: values.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const cmdEncoder = device.createCommandEncoder();
    cmdEncoder.copyBufferToBuffer(dataBuffer, 0, readBuffer, 0, values.byteLength);
    device.queue.submit([cmdEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const output = new Float32Array(readBuffer.getMappedRange());
    return output;
}

export function testBitonic() {
    // Usage example
    const values: Float32Array = new Float32Array(1_000_000);
    for (let i = 0; i < values.length; i++) {
        values[i] = Math.random();
    }
    
    // reference CPU sort
    const sorted = values.slice().sort((a, b) => a - b);
    console.log(sorted);

    bitonicSortWebGPU(values).then(sorted => {
        console.log(sorted);
    });
}