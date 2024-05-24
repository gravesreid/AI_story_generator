from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A5
from reportlab.lib.units import inch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, BlipProcessor, BlipForConditionalGeneration, BertTokenizer, BertModel
import soundfile as sf
from diffusers import DiffusionPipeline, ControlNetModel, AutoencoderKL, DDIMScheduler, StableDiffusionXLControlNetPipeline, LMSDiscreteScheduler, TCDScheduler
from huggingface_hub import hf_hub_download
import os
import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from supercat_characters import *

import io
from b2sdk.v1 import InMemoryAccountInfo, B2Api, UploadSourceBytes
# Backblaze stuff
info = InMemoryAccountInfo()
b2_api = B2Api(info)

with open('backblaze.txt', 'r') as backblaze:
    backblaze_key = backblaze.readline().strip()
with open('backblaze_id.txt', 'r') as backblaze_id:
    backblaze_id = backblaze_id.readline().strip()


b2_api.authorize_account("production", backblaze_id, backblaze_key)

bucket_name = 'dream-tails'
bucket = b2_api.get_bucket_by_name(bucket_name)

def upload_file_to_b2(bucket, folder_name, file_name, file_content):
    b2_file_name = os.path.join(folder_name, file_name)
    bucket.upload(UploadSourceBytes(file_content), b2_file_name)
    print(f"File '{file_name}' uploaded to '{b2_file_name}' in bucket '{bucket_name}'")


# end Backblaze stuff

def initialize_pipeline(device="cuda:0"):
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

def generate_image_prompts(story_sections, model, tokenizer, instructions, characters):
    # Create character instructions dynamically
    character_instructions = " ".join([f"If {char.get_name()} is in the scene include {char.get_description()}." for char in characters])
    full_instructions = instructions + " " + character_instructions
    
    img_prompts = []
    for section in story_sections:
        img_prompt_messages = [
            {"role": "system", "content": full_instructions},
            {"role": "user", "content": (section + ". only output the prompt, nothing else. Describe the character descriptions first, then the background. Limit to 70 words")},
        ]
        prompt = tokenizer.apply_chat_template(img_prompt_messages, tokenize=False, add_generation_prompt=False)
        img_prompt_output = model(prompt, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.6, top_p=0.9)
        sentences = img_prompt_output[0]["generated_text"][len(prompt):]
        sentences = sentences.split("\n\n")[1]  # Assuming first split is the valid prompt
        img_prompts.append(sentences)
    return img_prompts


def generate_images(img_prompts, pipe, folder_name):
    images = []
    for prompt in img_prompts:
        print(prompt)
        # get bert embeddings for prompt
        bert_prompt = bert_encode(prompt)
        result = pipe(prompt=("A picture of " + prompt), num_inference_steps=4, guidance_scale=0, eta=1)  # Adjust the parameters as needed
        image_path = os.path.join(folder_name, f"image_{len(images)}.png")
        result.images[0].save(image_path)
        # get blip description of generated image
        blip_caption = get_image_caption(image_path)
        print("Blip caption:", blip_caption)
        # get bert embeddings for blip caption
        bert_output = bert_encode(blip_caption)
        # convert bert embeddings to numpy arrays
        bert_prompt_np = bert_prompt.detach().cpu().numpy().mean(axis=0)
        bert_output_np = bert_output.detach().cpu().numpy().mean(axis=0)
        print("BERT prompt shape:", bert_prompt_np.shape)
        print("BERT output shape:", bert_output_np.shape)
        # calculate cosine similarity between prompt and output
        cos_sim = cosine_similarity([bert_prompt_np], [bert_output_np])[0][0]
        print("Cosine similarity:", cos_sim)
        images.append(image_path)
    return images

def get_image_caption(image_path):
    # Load the Blip model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

    # Load the image
    raw_image = Image.open(image_path)

    # Generate the image caption
    text = "a picture of"
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def bert_encode(text):
    # Load the BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    print("BERT output before squeeze:", outputs.last_hidden_state.shape)
    outputs = outputs.last_hidden_state.squeeze(0)
    print("BERT output after squeeze:", outputs.shape)

    return outputs

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

    # Save the PDF to backblaze
    with open(pdf_path, 'rb') as pdf_file:
        pdf_content = pdf_file.read()
        upload_file_to_b2(bucket, folder_name, filename, pdf_content)
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
        audio_file_path = os.path.join(folder_name, f"audio_section_{i}.wav")
        sf.write(audio_file_path, audio_arr, model.config.sampling_rate)
        audio_paths.append(audio_file_path)

        # Save audio to backblaze
        with open(audio_file_path, 'rb') as audio_file:
            audio_content = audio_file.read()
            upload_file_to_b2(bucket, folder_name, f"audio_section_{i}.wav", audio_content)
        
        print(f"Audio for section {i+1} saved to {audio_file_path}")
    return audio_paths



def generate_combined_audio(audio_files, folder_name):
    combined_audio_path = os.path.join(folder_name, "combined_audio.wav")
    combined_audio = []
    sample_rate = None

    for audio_file in audio_files:
        # Ensure the audio file exists
        if not os.path.exists(audio_file):
            print(f"Error: Audio file {audio_file} does not exist.")
            continue

        try:
            audio, sr = sf.read(audio_file)
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                raise ValueError(f"Sample rate mismatch: {audio_file} has {sr}, expected {sample_rate}")

            combined_audio.extend(audio)
        except Exception as e:
            print(f"Error reading {audio_file}: {e}")
            continue

    # Write the combined audio to a single file
    if combined_audio:
        combined_audio = np.array(combined_audio)
        sf.write(combined_audio_path, combined_audio, sample_rate)
        print(f"Combined audio saved to {combined_audio_path}")
    else:
        print("No audio data to combine.")
    
    # Save combined audio to backblaze
    with open(combined_audio_path, 'rb') as audio_file:
        combined_audio_content = audio_file.read()
        upload_file_to_b2(bucket, folder_name, "combined_audio.wav", combined_audio_content)
    
    return combined_audio_path

