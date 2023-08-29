// The bitonic sort algorithm itself is implemented in bitonic.ts
// but we need to do some extra work to sort the depth buffer:
// 1. compute the depth of each vertex
// 2. pad the depth buffer to the next power of 2
// 3. sort the depth buffer
// 4. copy the indices to the index buffer, replicating each index 6 times (since each quad has 6 vertices)
//
// This file implements steps 1, 2 and 4.

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

    // TODO: we're using the entire projection matrix and compute (x, y, z, w) of each
    // vertex, but we only need the z component. We could save some bandwidth by
    // only sending the z component of the projection matrix and only computing
    // the z component of each vertex.
    projMatrixBuffer: GPUBuffer; // projection matrix, set at each frame
    depthBuffer: GPUBuffer; // depth values, computed each time using uniforms, padded to next power of 2
    indexBuffer: GPUBuffer; // resulting index buffer, per vertex, 6 * #nElements

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

        // buffer for the vertex positions, set once
        this.positionsBuffer = this.context.device.createBuffer({
            size: gaussians.positionsArrayLayout.size,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
            label: "depthSorter.positionsBuffer"
        });
        new Uint8Array(this.positionsBuffer.getMappedRange()).set(new Uint8Array(gaussians.positionsBuffer));
        this.positionsBuffer.unmap();
        
        // buffer for the depth values, computed each time using uniforms, padded to next power of 2
        this.depthBuffer = this.context.device.createBuffer({
            size: this.nPadded * 4, // f32
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            label: "depthSorter.depthBuffer"
        });

        // buffer for the projection matrix, set at each frame
        this.projMatrixBuffer = this.context.device.createBuffer({
            size: projMatrixLayout.size,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "depthSorter.projMatrixBuffer"
        });

        // manually create the bind group layout because
        // this.computeDepthPipeline.getBindGroupLayout(0) doesn't work for some reason
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

        // buffer for the resulting index buffer, per vertex, 6 * #nElements
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

        this.sorter.destroy();
    }

    sort(projMatrix: number[][]): GPUBuffer {
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

            this.context.device.queue.submit([commandEncoder.finish()]);
        }

        return this.indexBuffer;
    }
}
