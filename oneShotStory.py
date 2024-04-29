
import transformers
import torch
from diffusers import DiffusionPipeline, DDIMScheduler
from huggingface_hub import hf_hub_download
from reportlab.lib.pagesizes import A5
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import pagesizes

from utils import generate_image_prompts, generate_images, create_pdf, split_story_into_sections, generate_audio

torch.cuda.empty_cache()

device = "cuda:1"

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

story_pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    token="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",
    device_map="auto",
)

story_messages = [
    {"role": "system", "content": "You are a an author of childrens books for ages 4 and under."},
    {"role": "user", "content": "Make a story about dinosaurs and their adventures."},
]

story_prompt = story_pipeline.tokenizer.apply_chat_template(
        story_messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    story_pipeline.tokenizer.eos_token_id,
    story_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

story_outputs = story_pipeline(
    story_prompt,
    max_new_tokens=1028,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(story_outputs[0]["generated_text"][len(story_prompt):])
story_text = story_outputs[0]["generated_text"][len(story_prompt):]

# Split the story into sections
story_sections = split_story_into_sections(story_text)

# generate audio for each page of the story
generate_audio(story_sections, device=device)


# generate images for each page of the story
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
# Take 2-steps lora as an example
ckpt_name = "Hyper-SDXL-4steps-lora.safetensors"
# Load model.
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to("cuda:1")
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora()
# Ensure ddim scheduler timestep spacing set as trailing !!!
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")


# Generate image prompts and images (functions from previous code)
img_prompts = generate_image_prompts(story_sections, story_pipeline, story_pipeline.tokenizer)
image_paths = generate_images(img_prompts, pipe)

# Create PDF with story sections and images
create_pdf(story_sections, image_paths)