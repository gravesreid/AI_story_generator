import torch
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, DDIMScheduler
from huggingface_hub import hf_hub_download

# Load original image
image = load_image("image_4.png")
image = np.array(image)
# Prepare Canny Control Image
low_threshold = 100
high_threshold = 200
image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
control_image = Image.fromarray(image)
control_image.save("control.png")
control_weight = 0.5  # recommended for good generalization

# Initialize pipeline
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16).to("cuda")

pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SDXL-8steps-lora.safetensors"))
# Ensure ddim scheduler timestep spacing set as trailing !!!
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe.fuse_lora()
image = pipe("A black Kitty fighting evil racoons", num_inference_steps=8, image=control_image, guidance_scale=0, controlnet_conditioning_scale=control_weight).images[0]
image.save('image_out.png')
