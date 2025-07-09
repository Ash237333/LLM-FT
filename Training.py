from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

import Model_Loader
from Eval import compute_metrics

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model, tokenizer = Model_Loader.load_model()

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
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
