from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A5
from reportlab.lib.units import inch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from diffusers import DiffusionPipeline, ControlNetModel, AutoencoderKL, DDIMScheduler, StableDiffusionXLControlNetPipeline, LMSDiscreteScheduler, TCDScheduler
from huggingface_hub import hf_hub_download
import os
import torch

from dyno_characters import *

def initialize_pipeline(device="cuda"):
        # Initial standard pipeline for the first image
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    repo_name = "ByteDance/Hyper-SD"
    # Take 2-steps lora as an example
    ckpt_name = "Hyper-SDXL-2steps-lora.safetensors"
    # Load model.
    pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(device)
    pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
    pipe.fuse_lora()
    # Ensure ddim scheduler timestep spacing set as trailing !!!
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    return pipe


def generate_image_prompts(story_sections, model, tokenizer):
    img_prompts = []
    for section in story_sections:
        img_prompt_messages = [
            {"role": "system", "content": 'You create short, minimal descriptions for an image generation model. For example, if a scene includes a cat that is sitting on a chair, the prompt would be "A black cat sitting on a chair". All the characters are dinosaurs. The dinosaurs are: Tina, a green Tyrannosaurus Rex; Trixie, an orange Triceratops; Vicky, a green velociraptor; and Benny, a grey Brachiosaurus.'},
            {"role": "user", "content": (section + ". only output the prompt, nothing else. Describe the dinosaur descriptions first, then the background. Limit to 70 words")},
        ]
        prompt = tokenizer.apply_chat_template(img_prompt_messages, tokenize=False, add_generation_prompt=False)
        img_prompt_output = model(prompt, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.6, top_p=0.9)
        sentences = img_prompt_output[0]["generated_text"][len(prompt):]
        print(sentences)
        sentences = sentences.split("\n\n")[1]  # Assuming first split is the valid prompt
        img_prompts.append(sentences)
    return img_prompts

def generate_images(img_prompts, pipe, folder_name):
    images = []
    for prompt in img_prompts:
        print(prompt)
        result = pipe(prompt=("A picture of " + prompt), num_inference_steps=4, guidance_scale=0, eta = 1)  # Adjust the parameters as needed
        image_path = os.path.join(folder_name, f"image_{len(images)}.png")
        result.images[0].save(image_path)
        images.append(image_path)
    return images

def create_pdf(story_sections, image_paths, folder_name, filename="output_story.pdf"):
    # Create a PDF file
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

def generate_audio(story_sections, folder_name, device="cuda"):
    # Load the TTS model and tokenizer
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

    audio_paths = []
    
    # Iterate through the sections of the story
    for i, section in enumerate(story_sections):
        description = "A female speaker with a slightly high-pitched voice delivers her words quite expressively, in a clear audio quality."
        
        # Tokenize text and description
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(section, return_tensors="pt").input_ids.to(device)
        
        # Generate audio
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        
        # Save audio to file
        audio_file_path = os.path.join(folder_name, f"audio_section_{i + 1}.wav")
        sf.write(audio_file_path, audio_arr, model.config.sampling_rate)
        audio_paths.append(audio_file_path)
        
        print(f"Audio for section {i+1} saved to {audio_file_path}")
    return audio_paths

