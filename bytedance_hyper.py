import torch
from diffusers import DiffusionPipeline, DDIMScheduler
from huggingface_hub import hf_hub_download
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
# Take 2-steps lora as an example
ckpt_name = "Hyper-SDXL-2steps-lora.safetensors"
# Load model.
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora()
# Ensure ddim scheduler timestep spacing set as trailing !!!
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
# lower eta results in more detail
prompt="Create an illustration of a heroic cat, named Whiskers, who has been gifted with the power of super speed, agility, and invisibility, and is about to embark on a mission to thwart the mischievous raccoon gang and bring peace back to Whiskerville."
image=pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0).images[0]
image.save("hyper_output.png")
