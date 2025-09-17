import os
from PIL import Image
import inferless
import torch
import cv2
from diffusers import ControlNetModel, AutoPipelineForImage2Image
from diffusers.utils import load_image
import numpy as np
import base64

import gc
from io import BytesIO

# Default prompts sourced from sandbox_og.ipynb
SYS_PROMPT = (
    "A stunning photograph of a newborn baby sleeping. "
    "(incredibly detailed, photorealistic skin texture:1.3), "
    "(perfectly formed face:1.2), closed eyes, peaceful expression. "
    "Lying in a soft, out-of-focus crib with warm blankets. "
    "(soft cinematic lighting:1.1), shallow depth of field, bokeh."
)

SYS_NEG_PROMPT = (
    "deformed, disfigured, (malformed:1.4), (extra limbs:1.3), extra fingers, "
    "fused fingers, misplaced limbs, artifacts, noise, blurry, grainy, "
    "weird textures, (anatomical errors:1.2), ugly"
)

app = inferless.Cls(gpu="A10")

class InferlessPythonModel:
    """
    Inferless model for SDXL Image2Image with Depth ControlNet.

    Inputs (see input_schema.py):
      - image_bytes: str URL or single-element list pointing to an image
      - inference_steps: int, diffusion steps (1..75)
      - controlet: float, ControlNet conditioning scale (0.0..2.0)
      - guidance: float, guidance scale (0.0..20.0)
      - prompt: str, optional; uses SYS_PROMPT if empty

    Output:
      - {"images": [base64_png_string]}
    """
    initialized = False
    @app.load
    def initialize(self):
        """Load ControlNet (depth SDXL) and the SDXL base pipeline on GPU.

        Env vars:
          - CONTROLNET_MODEL_ID: defaults to 'diffusers/controlnet-depth-sdxl-1.0'
          - BASE_MODEL_ID: defaults to 'stabilityai/stable-diffusion-xl-base-1.0'
          - IMAGE_SIZE: output resolution (default 1024)
          - ENABLE_OFFLOAD: enable CPU offload when 'true'
        """
        try:
            print("=== DEBUG: Starting model initialization ===")
            
            # Configure model IDs and settings from environment variables
            controlnet_id = os.getenv("CONTROLNET_MODEL_ID", "diffusers/controlnet-depth-sdxl-1.0")
            base_model_id = os.getenv("BASE_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
            self.image_size = int(os.getenv("IMAGE_SIZE", "1024"))
            enable_offload = True # os.getenv("ENABLE_OFFLOAD", "false").lower() in ("1", "true", "yes")

            print(f"DEBUG: Using ControlNet: {controlnet_id}")
            print(f"DEBUG: Using Base Model: {base_model_id}")
            print(f"DEBUG: Image size: {self.image_size}")
            print(f"DEBUG: Enable offload: {enable_offload}")

            # Load ControlNet (full variant) and base pipeline
            print("DEBUG: Loading ControlNet...")
            try:
                self.controlnet = ControlNetModel.from_pretrained(
                    controlnet_id,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True
                ).to("cuda")
                print("DEBUG: ControlNet loaded successfully")
            except Exception as e:
                print(f"ERROR: Failed to load ControlNet: {e}")
                raise

            print("DEBUG: Loading base pipeline...")
            try:
                self.pipeline = AutoPipelineForImage2Image.from_pretrained(
                    base_model_id,
                    controlnet=self.controlnet,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True
                )
                print("DEBUG: Base pipeline loaded successfully")
            except Exception as e:
                print(f"ERROR: Failed to load base pipeline: {e}")
                raise

            # Explicitly move to GPU, do not offload unless forced via env
            print("DEBUG: Configuring GPU/CPU offload...")
            if enable_offload:
                try:
                    self.pipeline.enable_model_cpu_offload()
                    print("DEBUG: CPU offload enabled successfully")
                except Exception as e:
                    print(f"DEBUG: CPU offload failed, moving to GPU: {e}")
                    # If offload is not available, proceed on GPU
                    self.pipeline.to("cuda")
                    print("DEBUG: Pipeline moved to GPU")
            else:
                self.pipeline.to("cuda")
                print("DEBUG: Pipeline moved to GPU")
                
            print("=== DEBUG: Model initialization completed successfully ===")
            
            initialized = True
        except Exception as e:
            print(f"CRITICAL ERROR during initialization: {e}")
            import traceback
            traceback.print_exc()
            raise



    @app.infer
    def infer(self, inputs):
        """Run i2i SDXL with Depth ControlNet.

        Inputs:
          - image_bytes: URL or list[str] length 1
          - inference_steps: int (clamped 1..75)
          - controlet: float ControlNet conditioning scale (0.0..2.0)
          - guidance: float guidance scale (0.0..20.0)
          - prompt: optional str; falls back to SYS_PROMPT

        Returns:
          dict(images=[base64 PNG])
        """
        try:
            print("=== DEBUG: Starting inference ===")
            print(f"Raw inputs received: {inputs}")
            
            # Helper to unwrap single-element lists from schema
            def _first(x):
                if isinstance(x, (list, tuple)) and len(x) > 0:
                    return x[0]
                return x

            # Extract and normalize inputs
            img_src = _first(inputs.get("image_bytes"))

            if img_src == "ping":

                if self.initialized:
                    return {
                    "images": "ON"
                    }
                else:
                    return {
                        "images": "OFF"
                    }

            print(f"DEBUG: Extracted image_bytes: {img_src}")
            
            if img_src is None:
                print("ERROR: image_bytes is None!")
                return {"images": []}
                
            num_inference_steps = int(_first(inputs.get("inference_steps", 24)))
            controlnet_conditioning_scale = float(_first(inputs.get("controlet", 0.65)))
            guidance = float(_first(inputs.get("guidance", 7.5)))
            prompt_in = _first(inputs.get("prompt", "")).strip() if inputs.get("prompt") is not None else ""

            print(f"DEBUG: Parameters - steps: {num_inference_steps}, controlnet_scale: {controlnet_conditioning_scale}, guidance: {guidance}")

            # Clamp ranges to safe values
            num_inference_steps = max(1, min(75, num_inference_steps))
            controlnet_conditioning_scale = max(0.0, min(2.0, controlnet_conditioning_scale))
            guidance = max(0.0, min(20.0, guidance))

            # Resolve prompts (default to system prompt if empty)
            prompt = SYS_PROMPT + prompt_in if prompt_in else SYS_PROMPT
            negative_prompt = SYS_NEG_PROMPT

            print(f"DEBUG: Final parameters - steps: {num_inference_steps}, controlnet_scale: {controlnet_conditioning_scale}, guidance: {guidance}")

            # Load and prepare images
            size = getattr(self, "image_size", 1024)
            print(f"DEBUG: Loading image from: {img_src}, target size: {size}")
            
            try:
                img = load_image(img_src).resize((size, size), Image.LANCZOS)
                print(f"DEBUG: Image loaded successfully, size: {img.size}")
            except Exception as e:
                print(f"ERROR: Failed to load image: {e}")
                return {"images": []}
            
            try:
                control_image = self.preprocess_img_improved(img, res=(size, size))
                print(f"DEBUG: Control image preprocessed successfully, size: {control_image.size}")
            except Exception as e:
                print(f"ERROR: Failed to preprocess control image: {e}")
                return {"images": []}

            output_images = []

            print(f"### Used prompt: {prompt}")
            print(f"### Used negative prompt: {negative_prompt}")
            print("-" * 10)

            print("DEBUG: Starting pipeline inference...")
            try:
                with torch.inference_mode():
                    tmp = self.pipeline(
                        image=img,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        control_image=control_image,
                        guidance_scale=float(guidance),
                        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
                        num_inference_steps=int(num_inference_steps),
                        height=size,
                        width=size
                    ).images
                    print(f"DEBUG: Pipeline generated {len(tmp)} images")
                    output_images.append(tmp[0])
                    del tmp
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"ERROR: Pipeline inference failed: {e}")
                import traceback
                traceback.print_exc()
                return {"images": []}

            # Convert to base64
            print(f"DEBUG: Converting {len(output_images)} images to base64...")
            try:
                output_images = [self.encode_base64(image) for image in output_images]
                print(f"DEBUG: Successfully encoded {len(output_images)} images")
            except Exception as e:
                print(f"ERROR: Failed to encode images to base64: {e}")
                import traceback
                traceback.print_exc()
                return {"images": []}
            
            # Try different return formats to match Inferless expectations
            result_v1 = {"images": output_images}
            result_v2 = {"generated_image": output_images[0] if output_images else ""}
            result_v3 = output_images
            
            print(f"DEBUG: Returning {len(output_images)} images")
            print(f"DEBUG: Return structure v1: {type(result_v1)}")
            print(f"DEBUG: Return keys v1: {list(result_v1.keys())}")
            print(f"DEBUG: First image length: {len(output_images[0]) if output_images else 0}")
            
            # Try the original format first
            return result_v1
            
        except Exception as e:
            print(f"CRITICAL ERROR in infer method: {e}")
            import traceback
            traceback.print_exc()
            return {"images": []}




    def finalize(self,args):
        """Release GPU memory and clear references."""
        self.pipeline = None
        self.controlnet = None
        self.initialized = False
        try:
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass



    def preprocess_img_improved(self, img: Image.Image, res: tuple[int, int] = (1024, 1024)):
        """Improved preprocessing: grayscale, resize, median blur, morphological opening, CLAHE, convert back to RGB."""
        # Convert to grayscale and resize
        img = img.convert("L").resize(res, Image.LANCZOS)
        img_np = np.array(img)

        # 1) Median blur to remove speckle noise
        img_np = cv2.medianBlur(img_np, 5)

        # 2) Morphological opening to eliminate small bright artifacts
        kernel = np.ones((5, 5), np.uint8)
        img_np = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel)

        # 3) CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_img = clahe.apply(img_np)

        # Convert to RGB for the pipeline input
        return Image.fromarray(contrast_img).convert("RGB")

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
