export class GpuContext {
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

        const device = await adapter.requestDevice({label: "GPUDevice"});

        return new GpuContext(gpu, adapter, device);
    }

    destroy() { 
        this.device.destroy();
        this.adapter = null as any;
        this.device = null as any;
    }
}
