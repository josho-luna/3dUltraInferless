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

    # Load Image and turn it into PIL image
    img = inputs["image_bytes"]
    prompt = [
        "A photo of a bald cute newborn baby, closed eyes." , 
       "Beautiful bald baby in his cradle with a blue baby cap", 
       "Beautiful bald baby", 

       "Portrait of a newborn baby with closed eyes, "
       "resting peacefully, with smooth and slightly pink skin, "
       "surrounded by a warm, soft environment reminiscent of the womb. "
       "Soft, diffuse lighting, focus on facial details, realistic skin textures and baby hands"
       ]
    negative_prompt = "Hair, deformed, fingers, sad, ugly, disgusting, uncanny"
    num_inference_steps = [10,15,12,20]
    controlnet_conditioning_scale = [0.6,0.7,0.65,0.67]
    guidance = [7,7.5,8,10]

    img = load_image(img).resize((1024, 1024), Image.LANCZOS)

    
    control_image = preprocess_img(img)


    output_imgs = []
    for i in range(4):
        output_image = self.pipeline(
            image = img,
            prompt=prompt[i],
            negative_prompt=negative_prompt,
            control_image=control_image, # The ultrasound image providing depth structure
            guidance_scale=guidance[i],
            controlnet_conditioning_scale=controlnet_conditioning_scale[i], 
            num_inference_steps=num_inference_steps[i],
            height=1024, # Specify output height
            width=1024,   # Specify output width
        ).images[0]
       
        buff = BytesIO()
        output_image.save(buff, format="PNG")
        img_str = base64.b64encode(buff.getvalue())
        output_imgs.append(img_str.decode('utf-8'))
  
    return {"images":output_imgs }
  

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