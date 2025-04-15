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

class InferlessPythonModel:
  
    def image_to_base64(image):
        buff = BytesIO()
        image.save(buff, format="PNG")
        return base64.b64encode(buff.getvalue()).decode("utf-8")

    @app.load
    def initialize(self):

        self.controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", # Depth ControlNet model
            torch_dtype=torch.float16,
            variant="fp16", 
            use_safetensors=True
        ).to("cuda")

        self.pipeline = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=self.controlnet,
            torch_dtype = torch.float16,
            variant = "fp16",

            use_safetensors=True
        ).to("cuda")



    @app.infer
    def infer(self, inputs):

        try:
            # Load Image and turn it into PIL image
            img = inputs["image_bytes"]

            img = load_image(img).resize((1024, 1024), Image.LANCZOS)
            control_image = self.preprocess_img(img)

            prompts = [
                "Photorealistic portrait of a bald cute asleep newborn baby, closed eyes, soft light, DSLR, 85mm lens",
            #   "Beautiful bald asleep newborn baby in a cradle, wearing a soft blue baby cap, warm diffuse lighting, natural tones",
                "Peaceful sleeping newborn baby, bald, close-up with detailed skin texture, photorealistic, pink skin, shallow depth of field",
                "Portrait of a sleeping newborn baby with closed eyes, resting peacefully in a soft womb-like environment, warm tones, detailed baby hands"
            ]

            negative_prompt = "hair, deformed, fingers, sad, ugly, disgusting, uncanny, blurry, grainy, monochrome, duplicate, artifact, watermark, text"

            num_inference_steps = [30, 25, 45]
            controlnet_conditioning_scale = [0.6, 0.75, 0.9]
            guidance = [7.0, 10.0, 12.0]

            # Optional: set different generators for reproducibility
            seeds = [43, 44, 45]
            generators = [torch.manual_seed(seed) for seed in seeds]

            # Run batch if your pipeline supports it
            with torch.inference_mode():
                output_images = self.pipeline(
                    image=[img] * 3,
                    prompt=prompts,
                    negative_prompt=[negative_prompt] * 3,
                    control_image=[control_image] * 3,
                    guidance_scale=guidance,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generators,
                    height=1024,
                    width=1024
                ).images

            # Convert to base64
            output_imgs = [self.image_to_base64(image) for image in output_images]
            return {"images": output_imgs}
        except Exception as e:
            print(e)
            return []
        



    def finalize(self,args):
        self.pipeline = None
        self.controlnet = None



    def preprocess_img(img: Image, res: tuple[int, int] = (1024, 1024)):
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

    