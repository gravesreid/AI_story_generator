import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    token="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",
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a llm that takes in a story description and generate a prompt for an image generator"},
    {"role": "user", "content": "The first adventure of our heroic unicorninjas begins on a sunny day in the mystical realm of Unicornia, where they reside. The group consists of five friends: Luna, the leader and most skilled in martial arts; Nova, the genius inventor and tech-whiz; Astra, the celestial navigator and astronomer; Zephyr, the wind-surfing master and thrill-seeker; and Lyra, the gentle healer and animal whisperer."},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])