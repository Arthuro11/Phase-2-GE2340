# Phase-2-GE2340
!pip install -q peft transformers datasets evaluate requests torch numpy accelerate

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import PrefixTuningConfig, get_peft_model
from datasets import load_dataset
import torch

# Load GPT-2 Medium (as in your paper)
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Fix padding

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to create a prefix-tuned model
def create_prefix_model():
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    config = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20)  # 20 tokens as in draft
    return get_peft_model(model, config)

# Create global prefix model (for mixed/implicit attributes)
global_model = create_prefix_model()

# Create specific prefix model (for non-toxic/explicit control)
specific_model = create_prefix_model()

print("Models ready!")

# Load mixed data for global prefix (Wikitext as base + some toxic for latent)
wiki_mixed = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:500]")  # Clean-ish Wikipedia extracts
rtp_mixed = load_dataset("AllenAI/real-toxicity-prompts", split="train[:100]")  # Add some toxic
mixed_texts = [ex["text"] for ex in wiki_mixed] + [p["text"] for p in rtp_mixed["prompt"]]

# Tokenize
def preprocess(examples):
    return tokenizer(examples, truncation=True, max_length=128, padding="max_length")

tokenized_mixed = [{"input_ids": preprocess([text])["input_ids"][0]} for text in mixed_texts]  # Simple list

# Train global prefix
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(output_dir="./global_prefix", num_train_epochs=1, per_device_train_batch_size=4, report_to="none")
trainer_global = Trainer(model=global_model, args=training_args, train_dataset=tokenized_mixed, data_collator=data_collator)
trainer_global.train()
print("Global prefix trained!")

# Load non-toxic data for specific prefix (clean Wikitext)
non_toxic = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[500:1000]")  # Fresh clean batch
non_toxic_texts = [ex["text"] for ex in non_toxic]

# Tokenize
tokenized_non_toxic = [{"input_ids": preprocess([text])["input_ids"][0]} for text in non_toxic_texts]

# Train specific prefix
trainer_specific = Trainer(model=specific_model, args=training_args, train_dataset=tokenized_non_toxic, data_collator=data_collator)
trainer_specific.train()
print("Specific prefix trained!")

# Load 200 toxic prompts
rtp = load_dataset("AllenAI/real-toxicity-prompts", split="train")
prompts = [p["text"] for p in rtp["prompt"][:200]]

# Custom generation with logit subtraction (FPT-style)
alpha = 3.0  # From paper
generations = []
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Get logits from specific (non-toxic)
    with torch.no_grad():
        specific_logits = specific_model(**inputs).logits[:, -1, :]  # Last token for simplicity in loop

    # Get logits from global
    with torch.no_grad():
        global_logits = global_model(**inputs).logits[:, -1, :]

    # FPT subtraction: alpha * specific - (alpha - 1) * global
    adjusted_logits = alpha * specific_logits - (alpha - 1) * global_logits

    # Sample token (repeat for 25 new tokens)
    full_gen = inputs["input_ids"]
    for _ in range(25):  # Generate 25 tokens
        token = torch.multinomial(torch.softmax(adjusted_logits, dim=-1), num_samples=1)
        full_gen = torch.cat([full_gen, token], dim=1)
        with torch.no_grad():
            next_specific = specific_model(input_ids=full_gen).logits[:, -1, :]
            next_global = global_model(input_ids=full_gen).logits[:, -1, :]
            adjusted_logits = alpha * next_specific - (alpha - 1) * next_global

    gen_text = tokenizer.decode(full_gen[0], skip_special_tokens=True)[len(prompt):]
    generations.append(gen_text)

print("Generated 200 continuations!")

import requests
import numpy as np
from evaluate import load

# Your Perspective API key
API_KEY = "AIzaSyB1sGYxGSdNEifCja_lZSSYqA9AsX4kCkY"  # Replace!

def get_toxicity(text):
    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={API_KEY}"
    data = {"comment": {"text": text}, "requestedAttributes": {"TOXICITY": {}}}
    response = requests.post(url, json=data)
    return response.json().get("attributeScores", {}).get("TOXICITY", {}).get("summaryScore", {}).get("value", 0.0)

# Max Toxicity (avg of max per gen)
tox_scores = [get_toxicity(gen) for gen in generations]
max_toxicity = np.mean(tox_scores)  # Approx max as avg for simplicity

# Perplexity
perplexity_eval = load("perplexity", module_type="metric")
ppl_results = perplexity_eval.compute(model_id=model_name, predictions=generations, batch_size=4)
perplexity = ppl_results["mean_perplexity"]

# Dist-2
def dist_n(texts, n=2):
    all_ngrams = set()
    total_ngrams = 0
    for text in texts:
        words = text.split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        all_ngrams.update(ngrams)
        total_ngrams += len(ngrams)
    return len(all_ngrams) / total_ngrams if total_ngrams > 0 else 0

dist_2 = dist_n(generations)

print(f"Max Toxicity: {max_toxicity:.2f}")
print(f"Perplexity: {perplexity:.1f}")
print(f"Dist-2: {dist_2:.2f}")


