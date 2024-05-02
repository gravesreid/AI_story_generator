from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A5
from reportlab.lib.units import inch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf


def generate_image_prompts(story_sections, model, tokenizer):
    img_prompts = []
    for section in story_sections:
        img_prompt_messages = [
            {"role": "system", "content": "You create short, descriptive prompts for an image model"},
            {"role": "user", "content": ("create a short prompt for an image of: " + section + ". only output the prompt, nothing else")},
        ]
        prompt = tokenizer.apply_chat_template(img_prompt_messages, tokenize=False, add_generation_prompt=False)
        img_prompt_output = model(prompt, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.6, top_p=0.9)
        sentences = img_prompt_output[0]["generated_text"][len(prompt):]
        sentences = sentences.split("\n\n")[1]  # Assuming first split is the valid prompt
        img_prompts.append(sentences)
    return img_prompts

def generate_images(img_prompts, pipe):
    images = []
    for prompt in img_prompts:
        print(prompt)
        result = pipe(prompt=("Modern animation of " + prompt), num_inference_steps=2, guidance_scale=0)  # Adjust the parameters as needed
        image_path = f"image_{len(images)}.png"
        result.images[0].save(image_path)
        images.append(image_path)
    return images

def create_pdf(story_sections, image_paths, filename="output_story.pdf"):
    # Create a PDF file
    c = canvas.Canvas(filename, pagesize=A5)
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

def split_story_into_sections(full_story_text):
    # This example assumes that paragraphs are separated by two newlines
    sections = full_story_text.strip().split('\n\n')
    return sections

def generate_audio(story_sections, device="cuda:1"):
    # Load the TTS model and tokenizer
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")
    
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
        audio_file_path = f"audio_section_{i+1}.wav"
        sf.write(audio_file_path, audio_arr, model.config.sampling_rate)
        
        print(f"Audio for section {i+1} saved to {audio_file_path}")

