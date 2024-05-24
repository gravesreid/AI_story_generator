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
import shutil

from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, DDIMScheduler
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image

from utilsbb import *
from dyno_characters import *



device = "cuda:0"

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
with open("dyno_seed.txt", "r") as f:
    for line in f:
        cleaned_line = line.strip()
    if cleaned_line:
        last_story += cleaned_line + " "
print(last_story)
n_stories = 20

characters = [tina, trixie, vicky, benny]
for i in range(n_stories):
    # delete previous directory
    if i > 0:
        prev_folder_name = f"dyno_tails_{i-1}"
        shutil.rmtree(prev_folder_name, ignore_errors=True)
        print(f"Deleted {prev_folder_name}")
    torch.cuda.empty_cache()
    folder_name = f"dyno_tails_{i}"
    os.makedirs(folder_name, exist_ok=True)  # Create the folder if it doesn't exist

    story_instructions = "You are an author of the dinosaur series. ".join([f"The protagonists in the series are: {char.get_name()}." for char in characters])

    story_messages = [
        {"role": "system", "content": story_instructions},
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
    print(audio_files)
    #audio_files = [os.path.join(folder_name, f) for f in audio_files]

    # make an audio file that combines all the audio files
    combined_audio = generate_combined_audio(audio_files, folder_name)



    # generate images for each page of the story
    eta = 0.5
    #standard_pipe, control_pipe = initialize_pipelines(device)
    instructions = "You create short, simple scene descriptions for an image generation model. The model doesn't know which characters are which, so describe the characters instead of including the character names."

    img_prompts = generate_image_prompts(story_sections, story_pipeline, story_pipeline.tokenizer, instructions, characters)
    # save image prompts to b2
    for i, img_prompt in enumerate(img_prompts):
        img_prompt_name = f"page_{i}_img_prompt.txt"
        img_prompt_byte_arr = io.BytesIO()
        img_prompt_byte_arr.write(img_prompt.encode())
        img_prompt_byte_arr.seek(0)
        img_prompt_content = img_prompt_byte_arr.getvalue()
        upload_file_to_b2(bucket, folder_name, img_prompt_name, img_prompt_content)
        print(f'saved img_prompt to b2: {img_prompt_name}')
    image_paths = generate_images(img_prompts, control_pipe, folder_name)

    # save images to b2
    for image_path in image_paths:
        image = os.path.basename(image_path)
        upload_file_to_b2(bucket, folder_name, image, open(image_path, 'rb').read())
        print(f'saved image to b2: {image_path}')
        
        


    # Create PDF with story sections and images
    pdf_path = create_pdf(story_sections, image_paths, folder_name)
    pdf_path = os.path.join(folder_name, pdf_path)