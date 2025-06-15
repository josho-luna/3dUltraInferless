import os
from PIL import Image
import inferless
import torch
import cv2
from diffusers import ControlNetModel, AutoPipelineForImage2Image
from diffusers.utils import load_image
import numpy as np
import base64
from io import BytesIO

app = inferless.Cls(gpu="A10")
RES = 720

class InferlessPythonModel:
    
    def encode_base64(self, image_rgb: Image, image_format: str = "PNG") -> str:
        """
        Encode a 3-channel RGB NumPy image to a base64 string.

        Parameters:
            image_rgb (np.ndarray): 3-channel RGB image.
            image_format (str): Image format to use ("PNG", "JPEG", etc.)

        Returns:
            str: Base64-encoded image string.
        """
        buffered = BytesIO()
        image_rgb.save(buffered, format=image_format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @app.load
    def initialize(self):

        # CHANGED: Switched to a smaller, faster ControlNet model.
        self.controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0-small", # Smaller Depth ControlNet model
            torch_dtype=torch.float16,
            variant="fp16", 
            use_safetensors=True
        ).to("cuda")

        # CHANGED: Switched to the faster SDXL-Turbo model.
        self.pipeline = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo", # Faster Turbo model
            controlnet=self.controlnet,
            torch_dtype = torch.float16,
            variant = "fp16",
            use_safetensors=True
        ).to("cuda")

    @app.infer
    def infer(self, inputs):

        # Load Image and turn it into PIL image
        img = inputs["image"]

        img = load_image(img).resize((RES, RES), Image.LANCZOS)
        control_image = self.preprocess_img(img)

        prompts = [
            "Photorealistic portrait of a bald cute asleep newborn baby, closed eyes, soft light, DSLR, 85mm lens",
            "Peaceful sleeping newborn baby, bald, close-up with detailed skin texture, photorealistic, pink skin, shallow depth of field",
            "Portrait of a sleeping newborn baby with closed eyes, resting peacefully in a soft womb-like environment, warm tones"
        ]

        negative_prompt = "hair, deformed, fingers, sad, ugly, disgusting, uncanny, blurry, grainy, monochrome, duplicate, artifact, watermark, text"

        # CHANGED: Drastically reduced inference steps, as required for SDXL-Turbo models for optimal speed.
        num_inference_steps = [2, 3, 4]
        controlnet_conditioning_scale = [0.6, 0.75, 0.7]
        # CHANGED: Guidance scale must be 0.0 for SDXL-Turbo.
        guidance = [0.0, 0.0, 0.0]

        # Optional: set different generators for reproducibility
        seeds = [43, 44, 45]
        generators = [torch.manual_seed(seed) for seed in seeds]


        output_imgs = []
        # Run batch if your pipeline supports it
        with torch.inference_mode():
            for i in range(len(prompts)):
                output_image = self.pipeline(
                    image=img,
                    prompt=prompts[i],
                    negative_prompt=negative_prompt,
                    control_image=control_image,
                    guidance_scale=guidance[i],
                    controlnet_conditioning_scale=controlnet_conditioning_scale[i],
                    # CHANGED: num_inference_steps for Turbo models should be very low
                    num_inference_steps=num_inference_steps[i],
                    generator=generators[i],
                    height=RES,
                    width=RES
                ).images[0]

                output_image = self.encode_base64(output_image)
                output_imgs.append(output_image)

                del output_image
                torch.cuda.empty_cache()

        return {"images": output_imgs}

    def finalize(self,args):
        self.pipeline = None
        self.controlnet = None
        torch.cuda.empty_cache()


    def preprocess_img(self, img: Image, res: tuple[int, int] = (RES, RES)):
        """Preprocesses the input image: Load, Grayscale, Resize, CLAHE, Denoise, Sharpen, Convert to RGB."""
        kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])


        try:
            # Load image, convert to grayscale, resize
            img = img.convert("L")

        except Exception as e:
            print(f"Error opening or processing image: {e}")
            return None
            
        img_np = np.array(img)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_img = clahe.apply(img_np)

        # Denoise the image first
        denoised_img = cv2.fastNlMeansDenoising(contrast_img, h=7) # Denoising strength
        
        # Sharpen the denoised image
        sharpened_img = cv2.filter2D(denoised_img, -1, kernel) 

        final = Image.fromarray(sharpened_img)
        # Convert final grayscale image to RGB for the pipeline input
        return final.convert("RGB")
