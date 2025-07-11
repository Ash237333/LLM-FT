import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

from Eval import compute_metrics
from Preprocessing import DataCollator, load_model, load_custom_dataset, prep_dataset

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model, tokenizer = load_model()

data_collator = DataCollator(tokenizer)

dataset = load_custom_dataset("Paper_Dataset.json", 0.2)
dataset = prep_dataset(dataset, tokenizer)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_arguments = TrainingArguments(
    output_dir="./lora",
    logging_dir="./logs",
    logging_steps=100,
    report_to="tensorboard",
    per_device_train_batch_size=256,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    eval_strategy="steps",
    eval_steps=500,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

torch.cuda.reset_peak_memory_stats()

trainer.train()
model.save_pretrained("./lora/final_adapter")

max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
print(f"\nMax GPU memory used: {max_memory:.2f} GB")
