from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
import torch
import time
import Model_Loader
from Eval import compute_metrics

def get_collate_fn(tokenizer):
    def collate_fn(batch):
        print("began putting into lists")
        input_ids = [example["input_ids"] for example in batch]
        attention_mask = [example["attention_mask"] for example in batch]
        labels = [example["labels"] for example in batch]

        print("began Padding")
        batch_encoding = tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt"
        )

        print("began padding labels")
        max_len = batch_encoding["input_ids"].shape[1]
        padded_labels = [label + [-100] * (max_len - len(label)) for label in labels]
        batch_encoding["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch_encoding
    return collate_fn

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model, tokenizer = Model_Loader.load_model()

data_collator = get_collate_fn(tokenizer)

dataset = Model_Loader.load_custom_dataset()
dataset = Model_Loader.prep_dataset(dataset, tokenizer)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_arguments = TrainingArguments(
    output_dir="./lora",
    logging_dir="./logs",
    logging_steps=100,
    report_to="tensorboard",
    per_device_train_batch_size=4,
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
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

torch.cuda.reset_peak_memory_stats()
start_time = time.time()

trainer.train()
model.save_pretrained("./lora/final_adapter")

end_time = time.time()
max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
print(f"\nMax GPU memory used: {max_memory:.2f} GB")
print(f"Training time: {(end_time - start_time)/60:.2f} minutes")
