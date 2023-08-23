import { mat4, vec3, mat3, Mat4, Mat3, Vec3 } from 'wgpu-matrix';

// camera as loaded from JSON
interface CameraRaw {
    id: number;
    img_name: string;
    width: number;
    height: number;
    position: number[];
    rotation: number[][];
    fx: number;
    fy: number;
}

// for some reason this needs to be a bit different than the one in wgpu-matrix
function getProjectionMatrix(znear: number, zfar: number, fovX: number, fovY: number): Mat4 {
    const tanHalfFovY: number = Math.tan(fovY / 2);
    const tanHalfFovX: number = Math.tan(fovX / 2);

    const top: number = tanHalfFovY * znear;
    const bottom: number = -top;
    const right: number = tanHalfFovX * znear;
    const left: number = -right;

    const P: Mat4 = mat4.create();

    const z_sign: number = 1.0;

    P[0] = (2.0 * znear) / (right - left);
    P[5] = (2.0 * znear) / (top - bottom);
    P[8] = (right + left) / (right - left);
    P[9] = (top + bottom) / (top - bottom);
    P[10] = z_sign * zfar / (zfar - znear);
    P[11] = -(zfar * znear) / (zfar - znear);
    P[14] = z_sign;
    P[15] = 0.0;

    return mat4.transpose(P);
}

// useful for coordinate flips
function diagonal4x4(x: number, y: number, z: number, w: number): Mat4 {
    const m = mat4.create();
    m[0] = x;
    m[5] = y;
    m[10] = z;
    m[15] = w;
    return m;
}

// A camera as used by the renderer. Interactivity is handled by InteractiveCamera.
export class Camera {
    height: number;
    width: number;
    viewMatrix: Mat4;
    perspective: Mat4;
    focalX: number;
    focalY: number;
    scaleModifier: number;

    constructor(
        height: number,
        width: number,
        viewMatrix: Mat4,
        perspective: Mat4,
        focalX: number,
        focalY: number,
        scaleModifier: number,
    ) {
        this.height = height;
        this.width = width;
        this.viewMatrix = viewMatrix;
        this.perspective = perspective;
        this.focalX = focalX;
        this.focalY = focalY;
        this.scaleModifier = scaleModifier;
    }

    static default(): Camera {
        return new Camera(
            500,
            500,
            mat4.lookAt([0, 0, 0], [0, 0, -1], [0, 1, 0]),
            mat4.perspective(Math.PI / 4, 1, 0.2, 100),
            600,
            600,
            1,
        )
    }

    // computes the depth of a point in camera space, for sorting
    dotZ(): (v: Vec3) => number {
        const depthAxis = this.depthAxis();
        return (v: Vec3) => {
            return vec3.dot(depthAxis, v);
        }
    }

    // gets the camera position in world space, for evaluating the spherical harmonics
    getPosition(): Vec3 {
        const inverseViewMatrix = mat4.inverse(this.viewMatrix);
        return mat4.getTranslation(inverseViewMatrix);
    }

    getProjMatrix(): Mat4 {
        var flippedY = mat4.clone(this.perspective);
        flippedY = mat4.mul(flippedY, diagonal4x4(1, -1, 1, 1));
        return mat4.multiply(flippedY, this.viewMatrix);
    }

    // for camera interactions
    translate(x: number, y: number, z: number) {
        const viewInv = mat4.inverse(this.viewMatrix);
        mat4.translate(viewInv, [x, y, z], viewInv);
        mat4.inverse(viewInv, this.viewMatrix);
    }

    // for camera interactions
    rotate(x: number, y: number, z: number) {
        const viewInv = mat4.inverse(this.viewMatrix);
        mat4.rotateX(viewInv, y, viewInv);
        mat4.rotateY(viewInv, x, viewInv);
        mat4.rotateZ(viewInv, z, viewInv);
        mat4.inverse(viewInv, this.viewMatrix);
    }

    // the depth axis is the third column of the transposed view matrix
    private depthAxis(): Vec3 {
        return mat4.getAxis(mat4.transpose(this.viewMatrix), 2);
    }
}

// Adds interactivity to a camera. The camera is modified by the user's mouse and keyboard input.
export class InteractiveCamera {
    private camera: Camera;
    private canvas: HTMLCanvasElement;

    private drag: boolean = false;
    private oldX: number = 0;
    private oldY: number = 0;
    private dRX: number = 0;
    private dRY: number = 0;
    private dRZ: number = 0;
    private dTX: number = 0;
    private dTY: number = 0;
    private dTZ: number = 0;

    private dirty: boolean = true;

    constructor(camera: Camera, canvas: HTMLCanvasElement) {
        this.camera = camera;
        this.canvas = canvas;

        this.createCallbacks();
    }

    static default(canvas: HTMLCanvasElement): InteractiveCamera {
        return new InteractiveCamera(Camera.default(), canvas);
    }

    private createCallbacks() {
        this.canvas.addEventListener('mousedown', (e) => {
            this.drag = true;
            this.oldX = e.pageX;
            this.oldY = e.pageY;
            this.setDirty();
            e.preventDefault();
        }, false);

        this.canvas.addEventListener('mouseup', (e) => {
            this.drag = false;
        }, false);

        this.canvas.addEventListener('mousemove', (e) => {
            if (!this.drag) return false;
            this.dRX = (e.pageX - this.oldX) * 2 * Math.PI / this.canvas.width;
            this.dRY = (e.pageY - this.oldY) * 2 * Math.PI / this.canvas.height;
            this.oldX = e.pageX;
            this.oldY = e.pageY;
            this.setDirty();
            e.preventDefault();
        }, false);

        this.canvas.addEventListener('wheel', (e) => {
            this.dTZ = e.deltaY * 0.1;
            this.setDirty();
            e.preventDefault();
        }, false);

        window.addEventListener('keydown', (e) => {
            const keyMap: {[key: string]: () => void} = {
                // translation
                'w': () => { this.dTY -= 0.1 },
                's': () => { this.dTY += 0.1 },
                'a': () => { this.dTX -= 0.1 },
                'd': () => { this.dTX += 0.1 },
                'q': () => { this.dTZ += 0.1 },
                'e': () => { this.dTZ -= 0.1 },

                // rotation
                'j': () => { this.dRX += 0.1 },
                'l': () => { this.dRX -= 0.1 },
                'i': () => { this.dRY += 0.1 },
                'k': () => { this.dRY -= 0.1 },
                'u': () => { this.dRZ += 0.1 },
                'o': () => { this.dRZ -= 0.1 },
            }

            if (!keyMap[e.key]) {
                return;
            } else {
                keyMap[e.key]();
                this.setDirty();
                e.preventDefault();
            }

        }, false);
    }

    public setNewCamera(newCamera: Camera) {
        this.camera = newCamera;
        this.setDirty();
    }

    private setDirty() {
        this.dirty = true;
    }
    
    private setClean() {
        this.dirty = false;
    }

    public isDirty(): boolean {
        return this.dirty;
    }

    public getCamera(): Camera {
        if (this.isDirty()) {
            this.camera.translate(this.dTX, this.dTY, this.dTZ);
            this.camera.rotate(this.dRX, this.dRY, this.dRZ);
            this.dTX = this.dTY = this.dTZ = this.dRX = this.dRY = this.dRZ = 0;
            this.setClean();
        }

        return this.camera;
    }
}

function focal2fov(focal: number, pixels: number): number {
    return 2 * Math.atan(pixels / (2 * focal));
}

function worldToCamFromRT(R: Mat3, t: Vec3): Mat4 {
    const R_ = R;
    const camToWorld = mat4.fromMat3(R_);
    const minusT = vec3.mulScalar(t, -1);
    mat4.translate(camToWorld, minusT, camToWorld);
    return camToWorld;
}

// converting camera coordinate systems is always black magic :(
function cameraFromJSON(rawCamera: CameraRaw, canvasW: number, canvasH: number): Camera {
    const fovX = focal2fov(rawCamera.fx, rawCamera.width);
    const fovY = focal2fov(rawCamera.fy, rawCamera.height);
    const projectionMatrix = getProjectionMatrix(0.2, 100, fovX, fovY);

    const R = mat3.create(...rawCamera.rotation.flat());
    const T = rawCamera.position;

    const viewMatrix = worldToCamFromRT(R, T);

    return new Camera(
        canvasH,
        canvasW,
        viewMatrix,
        projectionMatrix,
        rawCamera.fx,
        rawCamera.fy,
        Math.max(canvasW / rawCamera.width, canvasH / rawCamera.height),
    );
}

// A UI component that parses a JSON file containing a list of cameras and displays them as a list,
// allowing the user to choose from presets.
export class CameraFileParser {
    private fileInput: HTMLInputElement;
    private listElement: HTMLUListElement;
    private currentLineId: number = 0;
    private canvas: HTMLCanvasElement;
    private cameraSetCallback: (camera: Camera) => void;

    constructor(
        fileInput: HTMLInputElement,
        listElement: HTMLUListElement,
        canvas: HTMLCanvasElement,
        cameraSetCallback: (camera: Camera) => void,
    ) {
        this.fileInput = fileInput;
        this.listElement = listElement;
        this.canvas = canvas;
        this.cameraSetCallback = cameraSetCallback;

        this.fileInput.addEventListener('change', this.handleFileInputChange);
    }

    private handleFileInputChange = (event: Event) => {
        const file = this.fileInput.files?.[0];

        if (file) {
            const reader = new FileReader();
            reader.onload = this.handleFileLoad;
            reader.readAsText(file);
        }
    };

    private handleFileLoad = (event: ProgressEvent<FileReader>) => {
        if (!event.target) return;

        const contents = event.target.result as string;
        const jsonData = JSON.parse(contents);

        this.currentLineId = 0;
        this.listElement.innerHTML = '';

        jsonData.forEach((cameraJSON: any) => {
            this.currentLineId++;
            const listItem = document.createElement('li');
            const camera = cameraFromJSON(cameraJSON, this.canvas.width, this.canvas.height);
            listItem.textContent = cameraJSON.img_name;
            listItem.addEventListener('click', this.createCallbackForLine(camera));
            this.listElement.appendChild(listItem);
        });
    };

    private createCallbackForLine = (camera: Camera) => {
        return () => {
            this.cameraSetCallback(camera);
        };
    };
}