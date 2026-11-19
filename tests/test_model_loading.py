
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/dss/dsshome1/04/ru94wid2/models/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

model.eval()

messages = [
    {"role": "user", "content": "What is the capital of France?"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda:0")

with torch.inference_mode():
    output = model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=False,
    )

print(
    tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
)
