import os
import transformers
import torch
from diffusers import DiffusionPipeline, DDIMScheduler
from huggingface_hub import hf_hub_download
from reportlab.lib.pagesizes import A5
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import pagesizes
from transformers import pipeline

from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, DDIMScheduler
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image

from utils import generate_image_prompts, generate_images, create_pdf, split_story_into_sections, generate_audio, initialize_pipeline

device = "cuda:1"

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

with open('secret_token.txt', 'r') as secret_token:
    token = secret_token.readline().strip()


story_pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    token=token,
    device_map=device,
)
control_pipe = initialize_pipeline(device)
prompt_list = ['astronomy', 'biology', 'chemistry', 'geology', 'meteorology', 'oceanography', 'physics', 'zoology', 'botany', 'ecology', 'engineering']
file_names = [item.replace(" ", "_") for item in prompt_list]
for i, prompt in enumerate(prompt_list):
    torch.cuda.empty_cache()
    folder_name = file_names[i]
    os.makedirs(folder_name, exist_ok=True)  # Create the folder if it doesn't exist



    story_messages = [
        {"role": "system", "content": "You are a an author of childrens books for ages 3-6."},
        {"role": "user", "content": f"Write an educational book : {prompt}. Only output the text, nothing else."},
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
        max_new_tokens=2056,
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
    audio_files = generate_audio(story_sections, folder_name, device=device)
    audio_files = [os.path.join(folder_name, f) for f in audio_files]


    # generate images for each page of the story
    eta = 0.5
    #standard_pipe, control_pipe = initialize_pipelines(device)
    img_prompts = generate_image_prompts(story_sections, story_pipeline, story_pipeline.tokenizer)
    image_paths = generate_images(img_prompts, control_pipe, folder_name, eta=eta, device=device)
    del image_paths[0]
    #image_paths = [os.path.join(folder_name, path) for path in image_paths]
    #image_paths = generate_images(img_prompts, standard_pipe, control_pipe, eta=eta, device=device)
    print(image_paths)


    # Create PDF with story sections and images
    pdf_path = create_pdf(story_sections, image_paths, folder_name)
    pdf_path = os.path.join(folder_name, pdf_path)