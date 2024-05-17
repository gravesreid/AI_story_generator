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

from utils import *
from supercat_characters import *



device = "cuda:2"

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
last_story = ""
with open("seed_story.txt", "r") as f:
    for line in f:
        cleaned_line = line.strip()
    if cleaned_line:
        last_story += cleaned_line + " "
print(last_story)
n_stories = 15 
for i in range(n_stories):
    torch.cuda.empty_cache()
    folder_name = f"super_cat_volume_{i}"
    os.makedirs(folder_name, exist_ok=True)  # Create the folder if it doesn't exist



    story_messages = [
        {"role": "system", "content": "You are a an author of the Super Cat series. The protagonists in the series are: " + supercat.get_name() + " a " + supercat.get_gender() + supercat.get_description() + ", " + captainwhiskers.get_name() + " a " + captainwhiskers.get_gender() + captainwhiskers.get_description() + ". And the antagonists are:  " + professorcatnip.get_name() + " a " + professorcatnip.get_gender() + professorcatnip.get_description() + ", " + ladymeowington.get_name() + " a " + ladymeowington.get_gender() + ladymeowington.get_description() + "."},
        {"role": "user", "content": f"The last story was: {last_story}. Write the next story, with each page separated by an empty line. Only output the text, nothing else. Do not output image descriptions."},
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
        max_new_tokens=10000,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    print(story_outputs[0]["generated_text"][len(story_prompt):])
    story_text = story_outputs[0]["generated_text"][len(story_prompt):]

    # Split the story into sections
    story_sections = split_story_into_sections(story_text)
    new_story = ""
    for i,section in enumerate(story_sections):
        new_story += section + " "
        text_name = f"page_{i}.txt"
        text_byte_arr = io.BytesIO()
        text_byte_arr.write(section.encode())
        text_byte_arr.seek(0)
        text_content = text_byte_arr.getvalue()
        upload_file_to_b2(bucket, folder_name, text_name, text_content)
    last_story = new_story
        

    # generate audio for each page of the story
    audio_files = generate_audio(story_sections, folder_name, device=device)
    audio_files = [os.path.join(folder_name, f) for f in audio_files]


    # generate images for each page of the story
    eta = 0.5
    #standard_pipe, control_pipe = initialize_pipelines(device)
    img_prompts = generate_image_prompts(story_sections, story_pipeline, story_pipeline.tokenizer)
    image_paths = generate_images(img_prompts, control_pipe, folder_name)
    print(image_paths)


    # Create PDF with story sections and images
    pdf_path = create_pdf(story_sections, image_paths, folder_name)
    pdf_path = os.path.join(folder_name, pdf_path)