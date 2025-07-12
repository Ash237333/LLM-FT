import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from datetime import datetime

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

run_name = "8B_Instruct_B64_4_Paper" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

training_arguments = TrainingArguments(
    output_dir="./lora",
    logging_dir=f"./logs/{run_name}",
    logging_steps=100,
    report_to="tensorboard",
    per_device_train_batch_size=64,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=64,
    eval_do_concat_batches=False,
    eval_strategy="steps",
    eval_steps=5,
    fp16=True,
    eval_accumulation_steps=32,
    num_train_epochs=1
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
