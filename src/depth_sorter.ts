import { GpuContext } from './gpu_context';
import { BitonicSorter } from './bitonic';
import { PackedGaussians } from './ply';
import { f32, mat4x4 } from './packing';

function nextPowerOfTwo(x: number): number {
    return Math.pow(2, Math.ceil(Math.log2(x)));
}

// the depth of each vertex is computed, the excess space is padded with +inf
function computeDepthShader(itemsPerThread: number, numQuadsUnpadded: number): string {
    return `
@group(0) @binding(0) var<storage, read> vertices: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> depths: array<f32>;
@group(0) @binding(2) var<uniform> projMatrix: mat4x4<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    for (var i = global_id.x * ${itemsPerThread}; i < (global_id.x + 1) * ${itemsPerThread}; i++) {
        //if (i >= arrayLength(&vertices)) {
        if (i >= ${numQuadsUnpadded}) {
            depths[i] = 1e20f; // pad with +inf
        } else {
            let pos = vertices[i];
            let projPos = projMatrix * vec4<f32>(pos, 1.0);
            depths[i] = projPos.z;
        }
    }
}
`
}

// each quad index is repeated 6 times, once for each vertex
function copyToIndexBufferShader(itemsPerThread: number, numQuadsUnpadded: number): string {
    return `
@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(0) @binding(1) var<storage, read_write> indexBuffer: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    for (var i = global_id.x * ${itemsPerThread}; i < (global_id.x + 1) * ${itemsPerThread}; i++) {
        if (i >= ${numQuadsUnpadded}) {
            break;
        }
        let index = indices[i];

        for (var vertex = 0u; vertex < 6; vertex++) {
            indexBuffer[i * 6 + vertex] = index * 6 + vertex;
        }
    }
}
`
}

const projMatrixLayout = new mat4x4(f32);

export class DepthSorter {
    context: GpuContext;
    nUnpadded: number;
    nPadded: number;
    numThreads: number;

    positionsBuffer: GPUBuffer; // vertex positions, set once, #nElements
    projMatrixBuffer: GPUBuffer; // projection matrix, set at each frame
    depthBuffer: GPUBuffer; // depth values, computed each time using uniforms, padded to next power of 2
    indexBuffer: GPUBuffer; // resulting index buffer, per vertex, 6 * #nElements

    sortedReadoutBuffer: GPUBuffer; // readout of the sorted depth buffer for debugging, #nElements 
    depthReadoutBuffer: GPUBuffer; // readout of the depth buffer for debugging, #nElements

    computeDepthPipeline: GPUComputePipeline;
    computeDepthBindGroup: GPUBindGroup;

    copyToIndexBufferPipeline: GPUComputePipeline;
    copyToIndexBufferBindGroup: GPUBindGroup;

    sorter: BitonicSorter;

    constructor(context: GpuContext, gaussians: PackedGaussians) {
        this.context = context;
        this.nUnpadded = gaussians.numGaussians;
        this.nPadded = nextPowerOfTwo(this.nUnpadded);
        this.numThreads = 2048;

        this.positionsBuffer = this.context.device.createBuffer({
            size: gaussians.positionsArrayLayout.size,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
            label: "depthSorter.positionsBuffer"
        });
        new Uint8Array(this.positionsBuffer.getMappedRange()).set(new Uint8Array(gaussians.positionsBuffer));
        this.positionsBuffer.unmap();
        
        this.depthBuffer = this.context.device.createBuffer({
            size: this.nPadded * 4, // f32
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            label: "depthSorter.depthBuffer"
        });

        this.depthReadoutBuffer = this.context.device.createBuffer({
            size: this.nPadded * 4, // f32
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            label: "depthSorter.depthReadoutBuffer"
        });

        this.sortedReadoutBuffer = this.context.device.createBuffer({
            size: this.nPadded * 4, // u32
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            label: "depthSorter.sortedReadoutBuffer"
        });

        this.projMatrixBuffer = this.context.device.createBuffer({
            size: projMatrixLayout.size,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "depthSorter.projMatrixBuffer"
        });

        // manually create the bind group layout because this.computeDepthPipeline.getBindGroupLayout(0) doesn't work for some reason
        const computeDepthBindGroupLayout = this.context.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'read-only-storage' as GPUBufferBindingType,
                    },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'storage' as GPUBufferBindingType,
                    },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'uniform' as GPUBufferBindingType,
                    },
                },
            ],
        }); 

        const computeDepthPipelineLayout = this.context.device.createPipelineLayout({
            bindGroupLayouts: [computeDepthBindGroupLayout],
        });

        const paddedPerThread = Math.ceil(this.nPadded / this.numThreads);
        this.computeDepthPipeline = this.context.device.createComputePipeline({
            layout: computeDepthPipelineLayout, // would be easier to say layout: 'auto'
            compute: {
                module: this.context.device.createShaderModule({
                    code: computeDepthShader(paddedPerThread, this.nUnpadded),
                }),
                entryPoint: 'main',
            },
        });

        this.computeDepthBindGroup = this.context.device.createBindGroup({
            //layout: this.computeDepthPipeline.getBindGroupLayout(0), // this doesn't work for some reason
            layout: computeDepthBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.positionsBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.depthBuffer,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.projMatrixBuffer,
                    },
                },
            ],
        });

        this.sorter = new BitonicSorter(this.context, this.nPadded);

        this.indexBuffer = this.context.device.createBuffer({
            size: this.nUnpadded * 6 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            label: "depthSorter.indexBuffer"
        });

        const unpaddedPerThread = Math.ceil(this.nUnpadded / this.numThreads);
        this.copyToIndexBufferPipeline = this.context.device.createComputePipeline({
            compute: {
                module: this.context.device.createShaderModule({
                    code: copyToIndexBufferShader(unpaddedPerThread, this.nUnpadded),
                }),
                entryPoint: 'main',
            },
            layout: 'auto',
        });

        this.copyToIndexBufferBindGroup = this.context.device.createBindGroup({
            layout: this.copyToIndexBufferPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.sorter.indicesBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.indexBuffer,
                    },
                },
            ],
        });
    }

    destroy(): void {
        this.positionsBuffer.destroy();
        this.depthBuffer.destroy();
        this.projMatrixBuffer.destroy();
        this.indexBuffer.destroy();

        this.depthReadoutBuffer.destroy();
        this.sortedReadoutBuffer.destroy();

        this.sorter.destroy();
    }

    async sort(projMatrix: number[][]): Promise<GPUBuffer> {
        const projMatrixCpuBuffer = new ArrayBuffer(projMatrixLayout.size);
        projMatrixLayout.pack(0, projMatrix, new DataView(projMatrixCpuBuffer));

        this.context.device.queue.writeBuffer(
            this.projMatrixBuffer,
            0,
            projMatrixCpuBuffer,
            0,
            projMatrixCpuBuffer.byteLength
        );

        { // compute the depth of each vertex
            const commandEncoder = this.context.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.computeDepthPipeline);
            passEncoder.setBindGroup(0, this.computeDepthBindGroup);
            passEncoder.dispatchWorkgroups(this.numThreads);
            passEncoder.end();

            //commandEncoder.copyBufferToBuffer(
            //    this.depthBuffer,
            //    0,
            //    this.depthReadoutBuffer,
            //    0,
            //    this.nPadded * 4
            //);

            this.context.device.queue.submit([commandEncoder.finish()]);
        }

        // discard the result because we have already bound
        // this.sorter.indicesBuffer to the copyToIndexBufferBindGroup
        this.sorter.argsort(this.depthBuffer);


        { // copy the indices to the index buffer
            const commandEncoder = this.context.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.copyToIndexBufferPipeline);
            passEncoder.setBindGroup(0, this.copyToIndexBufferBindGroup);
            passEncoder.dispatchWorkgroups(this.numThreads);
            passEncoder.end();

            //commandEncoder.copyBufferToBuffer(
            //    this.sorter.indicesBuffer,
            //    0,
            //    this.sortedReadoutBuffer,
            //    0,
            //    this.nPadded * 4
            //);

            this.context.device.queue.submit([commandEncoder.finish()]);
        }

        //await this.depthReadoutBuffer.mapAsync(GPUMapMode.READ);
        //const depthReadoutArray = new Float32Array(this.depthReadoutBuffer.getMappedRange());
        //console.log('gpu depth', Array.from(depthReadoutArray));//.slice(0, this.nElements)));
        //this.depthReadoutBuffer.unmap();

        //await this.sortedReadoutBuffer.mapAsync(GPUMapMode.READ);
        //const sortedReadoutArray = new Uint32Array(this.sortedReadoutBuffer.getMappedRange());
        //console.log('gpu order', Array.from(sortedReadoutArray));//.slice(0, this.nElements)));
        //this.sortedReadoutBuffer.unmap();

        return this.indexBuffer;
    }
}
