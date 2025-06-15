import os
from PIL import Image
import inferless
import torch
import cv2
# ADDED: Import AutoencoderKL for the optimized VAE
from diffusers import ControlNetModel, AutoPipelineForImage2Image, AutoencoderKL 
from diffusers.utils import load_image
import numpy as np
import base64
from io import BytesIO

app = inferless.Cls(gpu="A10")

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

        # OPTIMIZATION: Load a faster, fp16-optimized VAE
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16
        )

        self.controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0-small",
            torch_dtype=torch.float16,
            variant="fp16", 
            use_safetensors=True
        ).to("cuda")

        self.pipeline = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            controlnet=self.controlnet,
            # OPTIMIZATION: Pass the faster VAE to the pipeline
            vae=vae,
            torch_dtype = torch.float16,
            variant = "fp16",
            use_safetensors=True
        ).to("cuda")
        
        # OPTIMIZATION: Enable xFormers for memory-efficient attention
        self.pipeline.enable_xformers_memory_efficient_attention()

        # OPTIMIZATION: Compile the UNet and VAE for a significant speed boost
        self.pipeline.unet = torch.compile(self.pipeline.unet, mode="reduce-overhead", fullgraph=True)
        self.pipeline.vae = torch.compile(self.pipeline.vae, mode="reduce-overhead", fullgraph=True)


    @app.infer
    def infer(self, inputs):

        # Load Image and turn it into PIL image
        img = inputs["image"]

        # CHANGED: Resized to 512x512 for optimal SDXL-Turbo performance
        img = load_image(img).resize((512, 512), Image.LANCZOS)
        control_image = self.preprocess_img(img, res=(512, 512))

        prompts = [
            "Photorealistic portrait of a bald cute asleep newborn baby, closed eyes, soft light, DSLR, 85mm lens",
            "Peaceful sleeping newborn baby, bald, close-up with detailed skin texture, photorealistic, pink skin, shallow depth of field",
            "Portrait of a sleeping newborn baby with closed eyes, resting peacefully in a soft womb-like environment, warm tones"
        ]

        negative_prompt = "hair, deformed, fingers, sad, ugly, disgusting, uncanny, blurry, grainy, monochrome, duplicate, artifact, watermark, text"
        
        num_inference_steps = [2, 3, 4]
        controlnet_conditioning_scale = [0.6, 0.75, 0.7]
        guidance = [0.0, 0.0, 0.0]

        seeds = [43, 44, 45]
        generators = [torch.manual_seed(seed) for seed in seeds]


        output_imgs = []
        with torch.inference_mode():
            # OPTIMIZATION: Warm-up run for torch.compile() - this makes subsequent runs faster
            # You can optionally remove this if the first generation's latency is not critical
            print("Performing a warm-up inference run...")
            _ = self.pipeline(
                prompt=prompts[0], 
                image=img, 
                control_image=control_image, 
                num_inference_steps=1, 
                guidance_scale=0.0,
                height=512,
                width=512
            )
            print("Warm-up complete.")

            for i in range(len(prompts)):
                output_image = self.pipeline(
                    image=img,
                    prompt=prompts[i],
                    negative_prompt=negative_prompt,
                    control_image=control_image,
                    guidance_scale=guidance[i],
                    controlnet_conditioning_scale=controlnet_conditioning_scale[i],
                    num_inference_steps=num_inference_steps[i],
                    generator=generators[i],
                    # CHANGED: Height and width set to 512 for optimal speed
                    height=512,
                    width=512
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


    # CHANGED: Default resolution is now 512x512
    def preprocess_img(self, img: Image, res: tuple[int, int] = (512, 512)):
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
