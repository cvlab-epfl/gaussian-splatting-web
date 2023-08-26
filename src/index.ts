import { loadFileAsArrayBuffer, PackedGaussians } from './ply';
import { CameraFileParser, InteractiveCamera } from './camera';
import { Renderer } from './renderer';
import { testBitonic } from './bitonic';

const canvas = document.getElementById("canvas-webgpu") as HTMLCanvasElement;

if (!navigator.gpu) {
    alert("WebGPU not supported on this browser! (navigator.gpu is null)");
}

let interactiveCamera = InteractiveCamera.default(canvas);

var currentRenderer: Renderer;

function handlePlyChange(event: any) {
    const file = event.target.files[0];
    const loadingPopup = document.getElementById('loading-popup')!;


    async function onFileLoad(arrayBuffer: ArrayBuffer) {
        if (currentRenderer) {
            await currentRenderer.destroy();
        }
        const gaussians = new PackedGaussians(arrayBuffer);
        try {
            const renderer = await Renderer.init(canvas, gaussians, interactiveCamera);
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
            .then(onFileLoad)
            .catch((error) => {
                alert(error);
            });
    }
}

//const plyFileInput = document.getElementById('plyButton');
//plyFileInput!.addEventListener('change', handlePlyChange);
//
//const cameraFileInput = document.getElementById('cameraButton')! as HTMLInputElement;
//const cameraList = document.getElementById('cameraList')! as HTMLUListElement;
//new CameraFileParser(cameraFileInput, cameraList, canvas, interactiveCamera.setNewCamera.bind(interactiveCamera));

testBitonic();