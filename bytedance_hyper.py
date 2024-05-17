import torch
from diffusers import DiffusionPipeline, TCDScheduler, DDIMScheduler
from huggingface_hub import hf_hub_download
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
# Take 2-steps lora as an example
ckpt_name = "Hyper-SDXL-2steps-lora.safetensors"
# Load model.
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to("cuda:1")
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora()
# Ensure ddim scheduler timestep spacing set as trailing !!!
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
# lower eta results in more detail
prompt="Tina, the green Tyrannosaurus Rex, Trixie, the orange Triceratops, Vicky, the green velociraptor, and Benny, the grey Brachiosaurus, stand together, ready for adventure. Tina's sharp teeth gleam, Trixie's frill shines, Vicky's sharp claws flex, and Benny's long neck stretches forward"
eta = 1
image=pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0, eta=eta).images[0]
image.save("hyper_output.png")
