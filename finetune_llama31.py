# Installing More Dependencies
import huggingface_hub
huggingface_hub.login("")
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import pandas as pd
import json
import os

model_id = "meta-llama/Meta-Llama-3.1-8B"

def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer

model, tokenizer = get_model_and_tokenizer(model_id)

# Function to format input and response for fine-tuning
def formatted_train(input, response) -> str:
    return f"<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"

# Load and prepare the dataset
def prepare_train_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    formatted_data = [{"prompt": k, "response": v} for k, v in data.items()]
    data_df = pd.DataFrame(formatted_data)
    
    # Create the text column in the appropriate format
    data_df["text"] = data_df.apply(lambda x: formatted_train(x["prompt"], x["response"]), axis=1)
    
    # Create a Hugging Face dataset from the DataFrame
    dataset = Dataset.from_pandas(data_df)
    return dataset

# Use the provided JSON file
json_file = "../workspace_data/protein_summaries.json"
train_dataset = prepare_train_data(json_file)

# LoRA configuration
peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

# Training arguments
training_arguments = TrainingArguments(
    output_dir="llama31_8B_protein",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=1,
    # max_steps=250,
    fp16=True,
    push_to_hub=True
)

# Trainer setup
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    args=training_arguments,
    tokenizer=tokenizer,
    packing=False,
    max_seq_length=1024
)

# Fine-tune the model
trainer.train()
