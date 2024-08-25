import base64
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np
import cv2
import requests
from motion_brush_utils import MotionBrush

app = FastAPI()

class AnimationInput(BaseModel):
    imageUrl: str
    mask: str  # Base64 encoded mask
    seed: int
    motion_bucket_id: int
    # New parameters
    num_inference_steps: int = Field(default=25)
    max_steps_to_replace: int = Field(default=25)
    num_frames: int = Field(default=25)
    fps: int = Field(default=8)
    noise_aug: float = Field(default=0.02)

class AnimationOutput(BaseModel):
    gif_base64: str

def download_image(url: str) -> Image.Image:
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download image")
    return Image.open(io.BytesIO(response.content))

def resize_image_and_mask_if_invalid(image, mask, max_size=576*1024, length_factor=64):
    height, width = image.shape[:2]

    if height * width > max_size:
        scale_factor = np.sqrt(max_size / (height * width))
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)

        new_height -= new_height % length_factor
        new_width -= new_width % length_factor

        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        print(f"1: Resized image from {height}x{width} to {new_height}x{new_width}")

        height, width = new_height, new_width

    if height % length_factor != 0 or width % length_factor != 0:
        new_height = height - height % length_factor
        new_width = width - width % length_factor

        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        print(f"2: Resized image from {height}x{width} to {new_height}x{new_width}")

    return image, mask

def animate_image(
    image,
    mask,
    num_inference_steps,
    max_steps_to_replace,
    num_frames,
    fps,
    motion_bucket_id,
    noise_aug,
    seed,
    max_height,
    max_width
):
    source_image = np.array(image.convert('RGBA'))
    mask_np = np.array(mask.convert('L'))

    # Extract the background (RGB) and mask
    source_image = source_image[..., :3]
    mask = (mask_np > 128).astype(np.float32)[..., np.newaxis]

    source_image, mask_np = resize_image_and_mask_if_invalid(source_image, mask_np, max_size=int(max_height)*int(max_width))

    if mask_np.sum() == 0:
        mask_np = None
    else:
        mask_np = (mask_np > 128).astype(np.float32)[..., np.newaxis]

    motion_brush = MotionBrush()

    gif_path = motion_brush(
        source_image, 
        mask_np,
        max_steps_to_replace=int(max_steps_to_replace),
        num_frames=int(num_frames),
        num_inference_steps=int(num_inference_steps),
        fps=int(fps),
        motion_bucket_id=int(motion_bucket_id),
        noise_aug_strength=float(noise_aug),
        seed=int(seed),
    )

    return gif_path

@app.post("/motion_brush", response_model=AnimationOutput)
async def create_animation(input_data: AnimationInput):
    try:
        # Download image from URL
        image = download_image(input_data.imageUrl)
        
        # Decode base64 mask
        mask_data = base64.b64decode(input_data.mask)
        mask = Image.open(io.BytesIO(mask_data))

        # Extract dimensions
        # max_width, max_height = map(int, input_data.dimensions.split('x'))
        max_width, max_height = image.size

        # Call animate_image function with all parameters
        output_gif_path = animate_image(
            image=image,
            mask=mask,
            num_inference_steps=input_data.num_inference_steps,
            max_steps_to_replace=input_data.max_steps_to_replace,
            num_frames=input_data.num_frames,
            fps=input_data.fps,
            motion_bucket_id=input_data.motion_bucket_id,
            noise_aug=input_data.noise_aug,
            seed=input_data.seed,
            max_height=max_height,
            max_width=max_width
        )

        # Read the GIF file and encode it to base64
        with open(output_gif_path, "rb") as gif_file:
            gif_data = gif_file.read()
            gif_base64 = base64.b64encode(gif_data).decode('utf-8')

        return AnimationOutput(gif_base64=gif_base64)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)