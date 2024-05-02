from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A5
from reportlab.lib.units import inch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os
import torch
from diffusers import DiffusionPipeline, ControlNetModel, AutoencoderKL, DDIMScheduler, StableDiffusionXLControlNetPipeline, LMSDiscreteScheduler, TCDScheduler
from huggingface_hub import hf_hub_download
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image


def generate_image_prompts(story_sections, model, tokenizer):

    img_prompts = []
    for section in story_sections:
        img_prompt_messages = [
            {"role": "system", "content": "You create short, descriptive prompts for an image model. Super cat is a black cat with green eyes and a yellow cape. He loves catnip. The model has no memory, so each prompt needs to contain information describing the characters physical description. The image prompt should be 70 words or less."},
            {"role": "user", "content": ("create a short prompt for an image of: " + section + " only output the prompt, nothing else")},
        ]
        prompt = tokenizer.apply_chat_template(img_prompt_messages, tokenize=False, add_generation_prompt=False)
        img_prompt_output = model(prompt, max_new_tokens=77, eos_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.6, top_p=0.9)
        sentences = img_prompt_output[0]["generated_text"][len(prompt):]
        sentences = sentences.split("\n\n")[1]  # Assuming first split is the valid prompt
        img_prompts.append(sentences)
    return img_prompts

def prepare_control_image(image_path, low_threshold=100, high_threshold=200):
    image = load_image(image_path)
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)
    control_image.save("control.png")
    return control_image

def initialize_pipelines(device="cuda"):
    # Initial standard pipeline for the first image
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    standard_pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16", scheduler=LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, num_train_timesteps=1000)).to(device)

    # ControlNet pipeline for subsequent images
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    control_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model_id, controlnet=controlnet, vae=vae, torch_dtype=torch.float16
    ).to(device)
    control_pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SDXL-2steps-lora.safetensors"))
    control_pipe.scheduler = TCDScheduler.from_config(control_pipe.scheduler.config)
    control_pipe.fuse_lora()

    return standard_pipe, control_pipe

def initialize_pipeline(device="cuda"):
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    control_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model_id, controlnet=controlnet, vae=vae, torch_dtype=torch.float16
    ).to(device)
    control_pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SDXL-2steps-lora.safetensors"))
    control_pipe.scheduler = TCDScheduler.from_config(control_pipe.scheduler.config)
    control_pipe.fuse_lora()
    
    return control_pipe


def generate_images(img_prompts, control_pipe, folder_name, control_weight=0.15, eta=0.25, device="cuda"):
    images = ["supercat.png"]  # Initially, no starting image; if needed, initialize with a specific image path.
    for idx, prompt in enumerate(img_prompts):
        if images:  # Prepare a control image if there are previous images
            control_image_path = images[-1]
            control_image = prepare_control_image(control_image_path)  # Ensure this function loads the image correctly
        else:
            control_image = None  # No control image for the first iteration

        print(f"Generating conditioned image for prompt: {prompt}")
        result = control_pipe(
            prompt="Modern animation of " + prompt,
            num_inference_steps=2,  # Assuming more steps might be needed; adjust as per model specifics
            image=control_image,
            guidance_scale=0,  # Adjusted for stronger adherence to the prompt, typical values might be higher
            controlnet_conditioning_scale=control_weight,
            eta=eta
        )

        # Save the generated image
        image_path = os.path.join(folder_name, f"image_{idx + 1}.png")  # Filename is now indexed based on prompt order
        result.images[0].save(image_path)
        images.append(image_path)

        print(f"Image saved to {image_path}")  # Confirm each save operation

    return images



def create_pdf(story_sections, image_paths, folder_name, filename="output_story.pdf"):
    # Create a PDF file path
    pdf_path = os.path.join(folder_name, filename)
    c = canvas.Canvas(pdf_path, pagesize=A5)
    width, height = A5

    # Ensure the image_paths list is as long as the story_sections list
    if len(image_paths) < len(story_sections):
        # Extend the list with None values if there are fewer images than text sections
        image_paths.extend([None] * (len(story_sections) - len(image_paths)))

    # Generate each page
    for text, image_path in zip(story_sections, image_paths):
        c.setFont("Helvetica", 12)

        # Insert image if exists
        if image_path:
            image_size = 4 * inch  # Size of the square image
            image_top = height - inch  # Top margin of 1 inch
            image_x = (width - image_size) / 2  # Center the image
            c.drawImage(image_path, image_x, image_top - image_size, width=image_size, height=image_size)

        # Create a text object for text below the image
        text_object = c.beginText(inch * 0.75, image_top - image_size - inch)
        text_object.setLeading(14)  # Line spacing

        # Split and add text ensuring it wraps within the width
        words = text.split()
        line = []
        max_width = width - 1.5 * inch  # Maximum width with margins
        for word in words:
            test_line = ' '.join(line + [word])
            if c.stringWidth(test_line, "Helvetica", 12) < max_width:
                line.append(word)
            else:
                text_object.textLine(' '.join(line))
                line = [word]
        text_object.textLine(' '.join(line))  # Add the last line

        c.drawText(text_object)  # Draw the text object
        c.showPage()

    # Save the PDF
    c.save()
    return pdf_path

def split_story_into_sections(full_story_text):
    # This example assumes that paragraphs are separated by two newlines
    sections = full_story_text.strip().split('\n\n')
    return sections

def generate_audio(story_sections, folder_name, device="cuda:0"):
    # Load the TTS model and tokenizer
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

    audio_paths = []
    
    # Iterate through the sections of the story
    for i, section in enumerate(story_sections):
        description = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a clear audio quality."
        
        # Tokenize text and description
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(section, return_tensors="pt").input_ids.to(device)
        
        # Generate audio
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        
        # Save audio to file
        audio_file_path = os.path.join(folder_name, f"audio_section_{i+1}.wav")
        sf.write(audio_file_path, audio_arr, model.config.sampling_rate)
        audio_paths.append(audio_file_path)
        
        print(f"Audio for section {i+1} saved to {audio_file_path}")
    
    return audio_paths

