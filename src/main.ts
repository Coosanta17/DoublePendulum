// Double Pendulum GPU Simulation using WGSL
import computeShader from './shaders/compute.wgsl?raw';
import vertexShader from './shaders/vertex.wgsl?raw';
import fragmentShader from './shaders/fragment.wgsl?raw';

const CANVAS_SIZE = 1024;
const SIMULATION_STEPS = 500; // Number of RK4 steps per pixel
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
    private simulationTime = 0;
    private animationId: number | null = null;

    constructor() {
        this.canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
        this.previewCanvas = document.getElementById('preview-canvas') as HTMLCanvasElement;
        this.previewInfo = document.getElementById('preview-info') as HTMLElement;
        this.previewCtx = this.previewCanvas.getContext('2d')!;

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

        // Select a random pixel initially
        this.selectRandomPixel();

        // Run the compute shader once
        this.runCompute();
        this.render();
    }

    private setupResources() {
        // Create result texture
        this.resultTexture = this.device.createTexture({
            size: [CANVAS_SIZE, CANVAS_SIZE, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });

        // Create uniform buffer
        const uniformData = new Float32Array([
            CANVAS_SIZE, CANVAS_SIZE, SIMULATION_STEPS, DT,
            M1, M2, L1, L2, G, 0, 0, 0 // padding for alignment to 48 bytes
        ]);

        this.uniformBuffer = this.device.createBuffer({
            size: 48, // round up to multiple of 16 for uniform buffer alignment
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        // Build an ArrayBuffer matching the WGSL Params layout: u32 width,height,steps; f32 dt,m1,m2,l1,l2,g; padding
        const uniformBuf = new ArrayBuffer(48);
        const dv = new DataView(uniformBuf);
        dv.setUint32(0, CANVAS_SIZE, true); // width
        dv.setUint32(4, CANVAS_SIZE, true); // height
        dv.setUint32(8, SIMULATION_STEPS, true); // steps
        dv.setFloat32(12, DT, true); // dt
        dv.setFloat32(16, M1, true);
        dv.setFloat32(20, M2, true);
        dv.setFloat32(24, L1, true);
        dv.setFloat32(28, L2, true);
        dv.setFloat32(32, G, true);
        // remaining 16 bytes are padding (zeros)
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformBuf);

        // Create sampler
        this.sampler = this.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });
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
            Math.ceil(CANVAS_SIZE / 8),
            Math.ceil(CANVAS_SIZE / 8)
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
        const bytesPerRow = CANVAS_SIZE * bytesPerPixel; // must be multiple of 256; 512*4=2048
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

    private async animatePreview(initialState: DoublePendulumState) {
        let state = {...initialState};
        const startTime = performance.now();

        const animate = () => {
            // RK4 step
            const k1 = this.derivatives(state);
            const k2 = this.derivatives({
                theta1: state.theta1 + 0.5 * DT * k1.dtheta1,
                theta2: state.theta2 + 0.5 * DT * k1.dtheta2,
                omega1: state.omega1 + 0.5 * DT * k1.domega1,
                omega2: state.omega2 + 0.5 * DT * k1.domega2,
            });
            const k3 = this.derivatives({
                theta1: state.theta1 + 0.5 * DT * k2.dtheta1,
                theta2: state.theta2 + 0.5 * DT * k2.dtheta2,
                omega1: state.omega1 + 0.5 * DT * k2.domega1,
                omega2: state.omega2 + 0.5 * DT * k2.domega2,
            });
            const k4 = this.derivatives({
                theta1: state.theta1 + DT * k3.dtheta1,
                theta2: state.theta2 + DT * k3.dtheta2,
                omega1: state.omega1 + DT * k3.domega1,
                omega2: state.omega2 + DT * k3.domega2,
            });

            state.theta1 += DT * (k1.dtheta1 + 2 * k2.dtheta1 + 2 * k3.dtheta1 + k4.dtheta1) / 6;
            state.theta2 += DT * (k1.dtheta2 + 2 * k2.dtheta2 + 2 * k3.dtheta2 + k4.dtheta2) / 6;
            state.omega1 += DT * (k1.domega1 + 2 * k2.domega1 + 2 * k3.domega1 + k4.domega1) / 6;
            state.omega2 += DT * (k1.domega2 + 2 * k2.domega2 + 2 * k3.domega2 + k4.domega2) / 6;

            this.drawPendulum(state);

            const elapsed = (performance.now() - startTime) / 1000;
            this.simulationTime = elapsed;
            document.getElementById('simTime')!.textContent = elapsed.toFixed(2);

            this.animationId = requestAnimationFrame(animate);
        };

        animate();
    }

    private derivatives(state: DoublePendulumState) {
        const delta = state.theta2 - state.theta1;
        const den1 = (M1 + M2) * L1 - M2 * L1 * Math.cos(delta) * Math.cos(delta);
        const den2 = (L2 / L1) * den1;

        const dtheta1 = state.omega1;
        const dtheta2 = state.omega2;

        const domega1 = (M2 * L1 * state.omega1 * state.omega1 * Math.sin(delta) * Math.cos(delta) +
            M2 * G * Math.sin(state.theta2) * Math.cos(delta) +
            M2 * L2 * state.omega2 * state.omega2 * Math.sin(delta) -
            (M1 + M2) * G * Math.sin(state.theta1)) / den1;

        const domega2 = (-M2 * L2 * state.omega2 * state.omega2 * Math.sin(delta) * Math.cos(delta) +
            (M1 + M2) * G * Math.sin(state.theta1) * Math.cos(delta) -
            (M1 + M2) * L1 * state.omega1 * state.omega1 * Math.sin(delta) -
            (M1 + M2) * G * Math.sin(state.theta2)) / den2;

        return {dtheta1, dtheta2, domega1, domega2};
    }

    private setupEventListeners() {
        this.canvas.addEventListener('click', async (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = Math.floor((e.clientX - rect.left) * CANVAS_SIZE / rect.width);
            const y = Math.floor((e.clientY - rect.top) * CANVAS_SIZE / rect.height);

            this.selectedPixel = {x, y};
            await this.updatePreview();
        });

        document.getElementById('reset')!.addEventListener('click', () => {
            this.selectRandomPixel();
        });
    }

    private selectRandomPixel() {
        const x = Math.floor(Math.random() * CANVAS_SIZE);
        const y = Math.floor(Math.random() * CANVAS_SIZE);
        this.selectedPixel = {x, y};
        this.updatePreview();
    }

    private async updatePreview() {
        if (!this.selectedPixel) return;

        if (this.animationId !== null) {
            cancelAnimationFrame(this.animationId);
        }

        this.previewInfo.style.display = 'none';
        this.simulationTime = 0;

        const initialState = await this.getPixelState(this.selectedPixel.x, this.selectedPixel.y);

        // Get the initial angles from pixel position
        const theta1_init = (this.selectedPixel.x / CANVAS_SIZE * 2.0 - 1.0) * Math.PI;
        const theta2_init = (this.selectedPixel.y / CANVAS_SIZE * 2.0 - 1.0) * Math.PI;

        // Start animation from initial position
        await this.animatePreview({
            theta1: theta1_init,
            theta2: theta2_init,
            omega1: 0,
            omega2: 0,
        });
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
