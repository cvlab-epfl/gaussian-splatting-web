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
import { InteractiveCamera } from './camera';
import { getShaderCode } from './shaders';
import { Mat4, Vec3 } from 'wgpu-matrix';
import { GpuContext } from './gpu_context';
import { DepthSorter } from './depth_sorter';

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
    numGaussians: number;

    context: GpuContext;
    contextGpu: GPUCanvasContext;

    uniformBuffer: GPUBuffer;
    pointDataBuffer: GPUBuffer;
    drawIndexBuffer: GPUBuffer;

    depthSorter: DepthSorter;

    uniformsBindGroup: GPUBindGroup;
    pointDataBindGroup: GPUBindGroup;

    drawPipeline: GPURenderPipeline;

    depthSortMatrix: number[][];

    destroyCallback: (() => void) | null = null;

    public static async requestContext(gaussians: PackedGaussians): Promise<GpuContext> {
        const gpu = navigator.gpu;
        if (!gpu) {
            return Promise.reject("WebGPU not supported on this browser! (navigator.gpu is null)");
        }

        const adapter = await gpu.requestAdapter();
        if (!adapter) {
            return Promise.reject("WebGPU not supported on this browser! (gpu.adapter is null)");
        }

        // for good measure, we request 1.5 times the amount of memory we need
        const byteLength = gaussians.gaussiansBuffer.byteLength;
        const device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: 1.5 * byteLength,
                maxBufferSize: 1.5 * byteLength,
            }
        });

        return new GpuContext(gpu, adapter, device);
    }

    // destroy the renderer and return a promise that resolves when it's done (after the next frame)
    public async destroy(): Promise<void> {
        return new Promise((resolve, reject) => {
            this.destroyCallback = resolve;
        });
    }

    constructor(
        canvas: HTMLCanvasElement,
        interactiveCamera: InteractiveCamera,
        gaussians: PackedGaussians,
        context: GpuContext,
    ) {
        this.canvas = canvas;
        this.interactiveCamera = interactiveCamera;
        this.context = context;
        const contextGpu = canvas.getContext("webgpu");
        if (!contextGpu) {
            throw new Error("WebGPU context not found!");
        }
        this.contextGpu = contextGpu;

        this.numGaussians = gaussians.numGaussians;

        const presentationFormat = "rgba16float" as GPUTextureFormat;

        this.contextGpu.configure({
            device: this.context.device,
            format: presentationFormat,
            alphaMode: 'premultiplied' as GPUCanvasAlphaMode,
        });

        this.pointDataBuffer = this.context.device.createBuffer({
            size: gaussians.gaussianArrayLayout.size,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
            label: "renderer.pointDataBuffer",
        });
        new Uint8Array(this.pointDataBuffer.getMappedRange()).set(new Uint8Array(gaussians.gaussiansBuffer));
        this.pointDataBuffer.unmap();

        // Create a GPU buffer for the uniform data.
        this.uniformBuffer = this.context.device.createBuffer({
            size: uniformLayout.size,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "renderer.uniformBuffer",
        });

        const shaderCode = getShaderCode(canvas, gaussians.sphericalHarmonicsDegree, gaussians.nShCoeffs);
        const shaderModule = this.context.device.createShaderModule({ code: shaderCode });

        this.drawPipeline = this.context.device.createRenderPipeline({
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

        this.uniformsBindGroup = this.context.device.createBindGroup({
            layout: this.drawPipeline.getBindGroupLayout(0),
            entries: [{
                binding: 0,
                resource: {
                    buffer: this.uniformBuffer,
                },
            }],
        });

        this.pointDataBindGroup = this.context.device.createBindGroup({
            layout: this.drawPipeline.getBindGroupLayout(1),
            entries: [{
                binding: 1,
                resource: {
                    buffer: this.pointDataBuffer,
                },
            }],
        });

        this.depthSorter = new DepthSorter(this.context, gaussians);

        this.drawIndexBuffer = this.context.device.createBuffer({
           size: 6 * 4 * gaussians.numGaussians,
           usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
           mappedAtCreation: false,
           label: "renderer.drawIndexBuffer",
        });

        // start the animation loop
        requestAnimationFrame(() => this.animate(true));
    }

    private destroyImpl(): void {
        if (this.destroyCallback === null) {
            throw new Error("destroyImpl called without destroyCallback set!");
        }

        this.uniformBuffer.destroy();
        this.pointDataBuffer.destroy();
        this.drawIndexBuffer.destroy();
        this.depthSorter.destroy();
        this.context.destroy();
        this.destroyCallback();
    }

    draw(nextFrameCallback: FrameRequestCallback): void {
        const commandEncoder = this.context.device.createCommandEncoder();

        // sort the draw order
        const indexBufferSrc = this.depthSorter.sort(this.depthSortMatrix);

        // copy the draw order to the draw index buffer
        commandEncoder.copyBufferToBuffer(
            indexBufferSrc,
            0,
            this.drawIndexBuffer,
            0,
            6 * 4 * this.depthSorter.nUnpadded,
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
        passEncoder.setPipeline(this.drawPipeline);

        passEncoder.setBindGroup(0, this.uniformsBindGroup);
        passEncoder.setBindGroup(1, this.pointDataBindGroup);

        passEncoder.setIndexBuffer(this.drawIndexBuffer, "uint32" as GPUIndexFormat)
        passEncoder.drawIndexed(this.numGaussians * 6, 1, 0, 0, 0);
        passEncoder.end();

        this.context.device.queue.submit([commandEncoder.finish()]);
        console.log('Drawn');

        requestAnimationFrame(nextFrameCallback);
    }

    animate(forceDraw?: boolean) {
        if (this.destroyCallback !== null) {
            this.destroyImpl();
            return;
        }
        if (!this.interactiveCamera.isDirty() && !forceDraw) {
            requestAnimationFrame(() => this.animate());
            return;
        }
        const camera = this.interactiveCamera.getCamera();

        const position = camera.getPosition();

        const tanHalfFovX = 0.5 * this.canvas.width / camera.focalX;
        const tanHalfFovY = 0.5 * this.canvas.height / camera.focalY;

        this.depthSortMatrix = mat4toArrayOfArrays(camera.viewMatrix);

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

        this.context.device.queue.writeBuffer(
            this.uniformBuffer,
            0,
            uniformsMatrixBuffer,
            0,
            uniformsMatrixBuffer.byteLength
        );

        this.draw(() => this.animate());
    }
}