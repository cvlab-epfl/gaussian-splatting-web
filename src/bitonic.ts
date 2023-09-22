// Implements a bitonic argsort on the GPU.
// It would be more efficient to use a radix sort, but this is simpler to implement.

import { GpuContext } from "./gpu_context";

const WGSIZE = 128;

function bitonicSortShader(itemsPerThread: number): string {
    return `
// Uniform buffer to store j and k
struct Uniforms {
    j: u32,
    k: u32,
};

@binding(0) @group(0) var<storage, read_write> data: array<vec4f>;
@binding(1) @group(0) var<storage, read_write> indices: array<vec4<u32>>;
@binding(2) @group(0) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(${WGSIZE})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = uniforms.j;
    let k = uniforms.k;
    
    let max = (global_id.x + 1) * ${itemsPerThread};
    for (var n = global_id.x * ${itemsPerThread}; n < max && (max ^ j) > max; n += 4) {
       var datan = data[n / 4];
       var indicesn = indices[n / 4];
       var swap = false;
       var dataixj = vec4f(0, 0, 0, 0);
       var indicesixj = vec4<u32>(0, 0, 0, 0);
       var lastIxjLoadedMajorIdx: u32 = 0;
       for (var l: u32 = 0; l < 4; l++) {
           let i = n + l;
           let datai = datan[l];
           let indicesi = indicesn[l];

           let ixj = i ^ j;
           if (ixj <= i) {
               continue;
           }

           let ixjMajor = ixj / 4;
           let ixjMinor = ixj % 4;
           let writeToWorkgroupBlock = (ixjMajor == n / 4);
           if (writeToWorkgroupBlock) {
               dataixj = datan;
               indicesixj = indicesn;
               lastIxjLoadedMajorIdx = 0;
           } else if (lastIxjLoadedMajorIdx != ixjMajor) {
               if (lastIxjLoadedMajorIdx != 0) {
                   data[lastIxjLoadedMajorIdx] = dataixj;
                   indices[lastIxjLoadedMajorIdx] = indicesixj;
               } 
               dataixj = data[ixjMajor];
               indicesixj = indices[ixjMajor];
               lastIxjLoadedMajorIdx = ixjMajor;
           }

           let swap_pos = ((i & k) == 0 && datai > dataixj[ixjMinor]);
           let swap_neg = ((i & k) != 0 && datai < dataixj[ixjMinor]);

           if (swap_pos || swap_neg) {
               swap = true;
               datan[l] = dataixj[ixjMinor];
               dataixj[ixjMinor] = datai;
               indicesn[l] = indicesixj[ixjMinor];
               indicesixj[ixjMinor] = indicesi;
               if (writeToWorkgroupBlock) {
                   datan[ixjMinor] = datai;
                   indicesn[ixjMinor] = indicesi;
               }
           }
       }
       if (swap) {
           data[n / 4] = datan;
           indices[n / 4] = indicesn;
       }
       if (lastIxjLoadedMajorIdx != 0) {
          data[lastIxjLoadedMajorIdx] = dataixj;
          indices[lastIxjLoadedMajorIdx] = indicesixj;
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
    bindGroups: GPUBindGroup[];

    valuesBuffer: GPUBuffer; // The buffer to sort
    indicesBuffer: GPUBuffer; // The indices of the values (to be sorted)
    initialIndexBuffer: GPUBuffer; // The initial indices of the values, to copy from at each call
    uniformBuffers: GPUBuffer[]; // The uniform buffers to use for each dispatch

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

        this.numThreads = 8192;
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

        this.createUniforms();
    }

    private createUniforms(): void {
        const gpuBuffers = [];
        const gpuBindGroups = [];

        for (let k = 2; k <= this.nElements; k <<= 1) {
            for (let j = k >> 1; j > 0; j = j >> 1) {
                const bufferContent = new Uint32Array([j, k]);
                const gpuBuffer = this.context.device.createBuffer({
                    size: bufferContent.byteLength,
                    usage: GPUBufferUsage.UNIFORM,
                    mappedAtCreation: true,
                    label: `bitonicSorter.uniformsBuffer.k=${k}.j=${j}`
                });
                new Uint32Array(gpuBuffer.getMappedRange()).set(bufferContent);
                gpuBuffer.unmap();

                const gpuBindGroup = this.context.device.createBindGroup({
                    layout: this.pipeline.getBindGroupLayout(0),
                    entries: [
                        {
                            binding: 0,
                            resource: {
                                buffer: this.valuesBuffer
                            },
                        },
                        {
                            binding: 1,
                            resource: {
                                buffer: this.indicesBuffer,
                            },
                        },
                        {
                            binding: 2,
                            resource: {
                                buffer: gpuBuffer
                            },
                        },
                    ],
                });

                gpuBuffers.push(gpuBuffer);
                gpuBindGroups.push(gpuBindGroup);
            }
        }

        this.uniformBuffers = gpuBuffers;
        this.bindGroups = gpuBindGroups;
    }

    public destroy(): void {
        this.valuesBuffer.destroy();
        this.indicesBuffer.destroy();
        this.initialIndexBuffer.destroy();
        for (const uniformBuffer of this.uniformBuffers) {
            uniformBuffer.destroy();
        }
    }

    argsort(values: GPUBuffer): GPUBuffer {
        if (values.size != this.valuesBuffer.size) {
            throw new Error("Input buffer size does not match the size of the sorter");
        }

        // Copy the data to the GPU
        const commandEncoder = this.context.device.createCommandEncoder();
        // clear just in case
        commandEncoder.clearBuffer(this.valuesBuffer);
        // copy the values
        commandEncoder.copyBufferToBuffer(values, 0, this.valuesBuffer, 0, values.size);

        // write the initial indices
        commandEncoder.copyBufferToBuffer(this.initialIndexBuffer, 0, this.indicesBuffer, 0, this.initialIndexBuffer.size);

        // Sort by dispatching the compute shader for each uniform buffer
        for (const uniformBindGroup of this.bindGroups) {
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.pipeline);
            passEncoder.setBindGroup(0, uniformBindGroup);
            passEncoder.dispatchWorkgroups(this.numThreads / WGSIZE);
            passEncoder.end();
        }

        this.context.device.queue.submit([commandEncoder.finish()]);

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