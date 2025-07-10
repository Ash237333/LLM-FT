import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import Eval
from Preprocessing import load_model, prep_dataset, load_custom_dataset, DataCollator

dataset = load_custom_dataset()
base_model, tokenizer = load_model()
dataset = prep_dataset(dataset, tokenizer)

data_collator = DataCollator(tokenizer)

eval_dataloader = DataLoader(
    dataset["test"],
    batch_size=4,
    collate_fn=data_collator
)

base_model.eval()

all_logits = []
all_labels = []

print("began eval")

with torch.no_grad():
    for batch in tqdm(eval_dataloader):
        input_ids = batch['input_ids'].to(base_model.device)
        attention_mask = batch['attention_mask'].to(base_model.device)
        labels = batch['labels'].to(base_model.device)

        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

all_logits = torch.cat(all_logits, dim=0)
all_labels = torch.cat(all_labels, dim=0)


class EvalPrediction:
    def __init__(self, predictions, label_ids):
        self.predictions = predictions
        self.label_ids = label_ids


eval_pred = EvalPrediction(predictions=all_logits, label_ids=all_labels)

results = Eval.compute_metrics(eval_pred, predict_threshold=True)

print("Evaluation results:", results)
