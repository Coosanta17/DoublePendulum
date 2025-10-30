// Double Pendulum GPU Simulation using WGSL
import computeShader from './shaders/compute.wgsl?raw';
import vertexShader from './shaders/vertex.wgsl?raw';
import fragmentShader from './shaders/fragment.wgsl?raw';

const CANVAS_SIZE = 1024;
const DT = 0.01; // Time step for RK4

// Physical parameters
const M1 = 1.0; // Mass of first pendulum
const M2 = 1.0; // Mass of second pendulum
const L1 = 1.0; // Length of first pendulum
const L2 = 1.0; // Length of second pendulum
const G = 9.81; // Gravity

interface DoublePendulumState {
    theta1: number;
    theta2: number;
    omega1: number;
    omega2: number;
}

class DoublePendulumSimulation {
    private device!: GPUDevice;
    private canvas: HTMLCanvasElement;
    private context!: GPUCanvasContext;
    private computePipeline!: GPUComputePipeline;
    private renderPipeline!: GPURenderPipeline;
    private resultTexture!: GPUTexture;
    private uniformBuffer!: GPUBuffer;
    private computeBindGroup!: GPUBindGroup;
    private renderBindGroup!: GPUBindGroup;
    private sampler!: GPUSampler;
    private previewCanvas: HTMLCanvasElement;
    private readonly previewCtx: CanvasRenderingContext2D;
    private selectedPixel: { x: number; y: number } | null = null;
    private previewInfo: HTMLElement;
    private previewCoords: HTMLElement;
    private simulationTime = 0;
    private animationId: number | null = null;

    constructor() {
        this.canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
        this.previewCanvas = document.getElementById('preview-canvas') as HTMLCanvasElement;
        this.previewInfo = document.getElementById('preview-info') as HTMLElement;
        this.previewCoords = document.getElementById('preview-coords') as HTMLElement;
        this.previewCtx = this.previewCanvas.getContext('2d')!;

        // Set canvas to fixed 1024x1024 size
        this.canvas.width = CANVAS_SIZE;
        this.canvas.height = CANVAS_SIZE;
        this.previewCanvas.width = 256;
        this.previewCanvas.height = 256;
    }

    async init() {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No GPU adapter found');
        }

        this.device = await adapter.requestDevice();

        this.context = this.canvas.getContext('webgpu')!;
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: format,
            alphaMode: 'opaque',
        });

        this.setupResources();
        this.setupPipelines(format);
        this.setupEventListeners();

        // Select center pixel initially
        this.selectedPixel = {
            x: Math.floor(CANVAS_SIZE / 2),
            y: Math.floor(CANVAS_SIZE / 2)
        };
        this.updatePreviewCoords();

        // Start animation loop
        this.startAnimation();
    }

    private setupResources() {
        // Create result texture
        this.resultTexture = this.device.createTexture({
            size: [CANVAS_SIZE, CANVAS_SIZE, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });

        // Create uniform buffer with time field
        this.uniformBuffer = this.device.createBuffer({
            size: 64, // increased for time field + padding
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Initialize with time = 0
        this.updateUniforms(0);

        // Create sampler
        this.sampler = this.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });
    }

    private updateUniforms(time: number) {
        // Build an ArrayBuffer matching the WGSL Params layout: u32 width,height,steps; f32 dt,m1,m2,l1,l2,g,time
        const uniformBuf = new ArrayBuffer(64);
        const dv = new DataView(uniformBuf);
        dv.setUint32(0, CANVAS_SIZE, true); // width
        dv.setUint32(4, CANVAS_SIZE, true); // height
        dv.setUint32(8, 0, true); // steps
        dv.setFloat32(12, DT, true); // dt
        dv.setFloat32(16, M1, true);
        dv.setFloat32(20, M2, true);
        dv.setFloat32(24, L1, true);
        dv.setFloat32(28, L2, true);
        dv.setFloat32(32, G, true);
        dv.setFloat32(36, time, true); // time
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformBuf);
    }

    private setupPipelines(format: GPUTextureFormat) {
        // Compute pipeline
        const computeModule = this.device.createShaderModule({
            code: computeShader,
        });

        const computeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {type: 'uniform'},
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        access: 'write-only',
                        format: 'rgba8unorm',
                    },
                },
            ],
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [computeBindGroupLayout],
            }),
            compute: {
                module: computeModule,
                entryPoint: 'main',
            },
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: computeBindGroupLayout,
            entries: [
                {binding: 0, resource: {buffer: this.uniformBuffer}},
                {binding: 1, resource: this.resultTexture.createView()},
            ],
        });

        // Render pipeline
        const vertexModule = this.device.createShaderModule({
            code: vertexShader,
        });

        const fragmentModule = this.device.createShaderModule({
            code: fragmentShader,
        });

        const renderBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: {sampleType: 'float'},
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: {},
                },
            ],
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [renderBindGroupLayout],
            }),
            vertex: {
                module: vertexModule,
                entryPoint: 'main',
            },
            fragment: {
                module: fragmentModule,
                entryPoint: 'main',
                targets: [{format}],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: renderBindGroupLayout,
            entries: [
                {binding: 0, resource: this.resultTexture.createView()},
                {binding: 1, resource: this.sampler},
            ],
        });
    }

    private runCompute() {
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();

        passEncoder.setPipeline(this.computePipeline);
        passEncoder.setBindGroup(0, this.computeBindGroup);
        passEncoder.dispatchWorkgroups(
            Math.ceil(CANVAS_SIZE / 16),
            Math.ceil(CANVAS_SIZE / 16)
        );
        passEncoder.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    private render() {
        const commandEncoder = this.device.createCommandEncoder();
        const textureView = this.context.getCurrentTexture().createView();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [
                {
                    view: textureView,
                    clearValue: {r: 0, g: 0, b: 0, a: 1},
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
        });

        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.draw(6);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    private async getPixelState(x: number, y: number): Promise<DoublePendulumState> {
        // Read pixel data from GPU
        const bytesPerPixel = 4; // rgba8unorm
        const bytesPerRow = CANVAS_SIZE * bytesPerPixel; // must be multiple of 256; 1024*4=4096
        const bufferSize = bytesPerRow * CANVAS_SIZE;
        const buffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyTextureToBuffer(
            {texture: this.resultTexture},
            {buffer, bytesPerRow},
            [CANVAS_SIZE, CANVAS_SIZE, 1]
        );
        this.device.queue.submit([commandEncoder.finish()]);

        await buffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = buffer.getMappedRange();
        const bytes = new Uint8Array(arrayBuffer);
        const rowStart = y * bytesPerRow;
        const pixelStart = rowStart + x * bytesPerPixel;
        const r = bytes[pixelStart] / 255.0;
        const g = bytes[pixelStart + 1] / 255.0;
        const b = bytes[pixelStart + 2] / 255.0;
        const a = bytes[pixelStart + 3] / 255.0;

        const state = {
            theta1: (r * 2.0 - 1.0) * Math.PI,
            theta2: (g * 2.0 - 1.0) * Math.PI,
            omega1: (b * 2.0 - 1.0) * 10.0,
            omega2: (a * 2.0 - 1.0) * 10.0,
        };

        buffer.unmap();
        buffer.destroy();

        return state;
    }

    private drawPendulum(state: DoublePendulumState) {
        const ctx = this.previewCtx;
        const w = this.previewCanvas.width;
        const h = this.previewCanvas.height;
        const scale = Math.min(w, h) / 4;
        const cx = w / 2;
        const cy = h / 2;

        ctx.clearRect(0, 0, w, h);

        // Draw pendulum
        const x1 = cx + L1 * scale * Math.sin(state.theta1);
        const y1 = cy + L1 * scale * Math.cos(state.theta1);
        const x2 = x1 + L2 * scale * Math.sin(state.theta2);
        const y2 = y1 + L2 * scale * Math.cos(state.theta2);

        // Rod 1
        ctx.strokeStyle = '#888';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(x1, y1);
        ctx.stroke();

        // Rod 2
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

        // Mass 1
        ctx.fillStyle = '#ff6b6b';
        ctx.beginPath();
        ctx.arc(x1, y1, 8, 0, Math.PI * 2);
        ctx.fill();

        // Mass 2
        ctx.fillStyle = '#4ecdc4';
        ctx.beginPath();
        ctx.arc(x2, y2, 8, 0, Math.PI * 2);
        ctx.fill();

        // Origin
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, Math.PI * 2);
        ctx.fill();
    }

    private async updatePreviewFromGPU() {
        if (!this.selectedPixel) return;

        const state = await this.getPixelState(this.selectedPixel.x, this.selectedPixel.y);
        this.drawPendulum(state);
    }

    private startAnimation() {
        const startTime = performance.now();

        const animate = async () => {
            const elapsed = (performance.now() - startTime) / 1000;
            this.simulationTime = elapsed;

            // Update uniforms with current time
            this.updateUniforms(elapsed);

            // Run compute shader
            this.runCompute();

            // Render to canvas
            this.render();

            // Update preview from GPU state
            if (this.selectedPixel) {
                await this.updatePreviewFromGPU();
            }

            // Update UI
            document.getElementById('simTime')!.textContent = elapsed.toFixed(2);

            this.animationId = requestAnimationFrame(animate);
        };

        animate();
    }

    private setupEventListeners() {
        this.canvas.addEventListener('click', async (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = Math.floor((e.clientX - rect.left) * CANVAS_SIZE / rect.width);
            const y = Math.floor((e.clientY - rect.top) * CANVAS_SIZE / rect.height);

            this.selectedPixel = {x, y};
            this.updatePreviewCoords();
        });

        document.getElementById('reset')!.addEventListener('click', () => {
            this.resetSimulation();
        });
    }

    private updatePreviewCoords() {
        if (!this.selectedPixel) {
            this.previewCoords.style.display = 'none';
            return;
        }

        this.previewCoords.style.display = 'block';
        this.previewCoords.textContent = `(${this.selectedPixel.x}, ${this.selectedPixel.y})`;
        this.previewInfo.style.display = 'none';
    }

    private resetSimulation() {
        // Reset simulation time but keep selected pixel
        this.simulationTime = 0;
        this.updateUniforms(0);
    }
}

(async () => {
    try {
        const sim = new DoublePendulumSimulation();
        await sim.init();
    } catch (error) {
        console.error('Failed to initialize:', error);
        alert('WebGPU initialization failed. Make sure your browser supports WebGPU.');
    }
})();
