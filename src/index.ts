import { loadFileAsArrayBuffer, PackedGaussians } from './ply';
import { CameraFileParser, InteractiveCamera } from './camera';
import { Renderer } from './renderer';

if (!navigator.gpu) {
    alert("WebGPU not supported on this browser! (navigator.gpu is null)");
}

// grab the DOM elements
const canvas = document.getElementById("canvas-webgpu") as HTMLCanvasElement;
const loadingPopup = document.getElementById('loading-popup')! as HTMLDivElement;
const fpsCounter = document.getElementById('fps-counter')! as HTMLLabelElement;
const cameraFileInput = document.getElementById('cameraButton')! as HTMLInputElement;
const cameraList = document.getElementById('cameraList')! as HTMLUListElement;
const plyFileInput = document.getElementById('plyButton') as HTMLInputElement;

// create the camera and renderer globals
let interactiveCamera = InteractiveCamera.default(canvas);
var currentRenderer: Renderer;

// swap the renderer when the ply file changes
function handlePlyChange(event: any) {
    const file = event.target.files[0];

    async function onFileLoad(arrayBuffer: ArrayBuffer) {
        if (currentRenderer) {
            await currentRenderer.destroy();
        }
        const gaussians = new PackedGaussians(arrayBuffer);
        try {
            const context = await Renderer.requestContext(gaussians);
            const renderer = new Renderer(canvas, interactiveCamera, gaussians, context, fpsCounter);
            currentRenderer = renderer; // bind to the global scope
            loadingPopup.style.display = 'none'; // hide loading popup
        } catch (error) {
            loadingPopup.style.display = 'none'; // hide loading popup
            alert(error);
        }
    }

    if (file) {
        loadingPopup.style.display = 'block'; // show loading popup
        loadFileAsArrayBuffer(file)
            .then(onFileLoad);
    }
}

// loads the default ply file (bundled with the source) at startup, useful for dev
async function loadDefaultPly() {
    const url = "pc_short.ply";
    loadingPopup.style.display = 'block'; // show loading popup
    const content = await fetch(url);
    const arrayBuffer = await content.arrayBuffer();
    const gaussians = new PackedGaussians(arrayBuffer);
    const context = await Renderer.requestContext(gaussians);
    const renderer = new Renderer(canvas, interactiveCamera, gaussians, context, fpsCounter);
    currentRenderer = renderer; // bind to the global scope
    loadingPopup.style.display = 'none'; // hide loading popup
}

// DEV: uncomment this line to load the default ply file at startup
//loadDefaultPly();

// add event listeners
plyFileInput!.addEventListener('change', handlePlyChange);
new CameraFileParser(
    cameraFileInput,
    cameraList,
    canvas,
    (camera) => interactiveCamera.setNewCamera(camera),
);