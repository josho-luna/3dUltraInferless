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
  

    @app.load
    def initialize(self):

        self.controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0-small", # Smaller Depth ControlNet model
            torch_dtype=torch.float16,
            variant="fp16", 
            use_safetensors=True
        ).to("cuda")

        self.pipeline = AutoPipelineForImage2Image.from_pretrained(
            
            #"stabilityai/sdxl-turbo", # Faster Turbo model
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=self.controlnet,
            torch_dtype = torch.float16,
            variant = "fp16",
            use_safetensors=True
        ).to("cuda")



    @app.infer
    def infer(self, inputs):

        
        # Load Image and turn it into PIL image
        img = inputs["image_bytes"]
        num_inference_steps = inputs['inference_steps']
        controlnet_conditioning_scale = inputs['controlet']
        guidance = inputs['guidance']
        prompt = inputs['prompt']


        img = load_image(img).resize((1024, 1024), Image.LANCZOS)
        control_image = self.preprocess_img(img)

        

        negative_prompt = "hair, deformed, fingers, sad, ugly, disgusting, uncanny, blurry, grainy, monochrome, duplicate, artifact, watermark, text"


        # Optional: set different generators for reproducibility
        seeds = [43, 44, 45]
        generators = [torch.manual_seed(seed) for seed in seeds]

        # Run batch if your pipeline supports it
        output_images = []

        

        with torch.inference_mode():
            tmp = self.pipeline(
                image=img,
                prompt= prompt,
                negative_prompt=negative_prompt,
                control_image=control_image,
                guidance_scale=float(guidance),
                controlnet_conditioning_scale=float(controlnet_conditioning_scale),
                num_inference_steps=int(num_inference_steps),
                height=1024,
                width=1024
            ).images
            output_images.append(tmp[0])
            del tmp
            torch.cuda.empty_cache()
            
                

        # Convert to base64
        output_images = [self.encode_base64(image) for image in output_images]
        return {"images": output_images}




    def finalize(self,args):
        self.pipeline = None
        self.controlnet = None



    def preprocess_img(self, img: Image, res: tuple[int, int] = (1024, 1024)):

        """Preprocesses the input image: Load, Grayscale, Resize, CLAHE, Denoise, Sharpen, Convert to RGB."""
        kernel = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])


        img = img.convert("L")
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

    def encode_base64(self, image: Image.Image, image_format: str = "PNG") -> str:
        """
        Encode a PIL image to a base64 string.

        Parameters:
            image (Image.Image): The PIL image to encode.
            image_format (str): Image format to use ("PNG", "JPEG", etc.)

        Returns:
            str: Base64-encoded image string.
        """
        buffered = BytesIO()
        image.save(buffered, format=image_format)

        img64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        if img64 is None:
            raise TypeError("Please check this goddam function")
        
        return img64 
