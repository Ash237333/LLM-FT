import torch
from torch.utils.data import DataLoader
import Model_Loader, Eval
from tqdm import tqdm

dataset = Model_Loader.load_custom_dataset()
base_model, tokenizer = Model_Loader.load_model()
dataset = Model_Loader.prep_dataset(dataset, tokenizer)


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

eval_dataloader = DataLoader(
    dataset["test"],
    batch_size=4,
    collate_fn=collate_fn
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

# Concatenate batches
all_logits = torch.cat(all_logits, dim=0)
all_labels = torch.cat(all_labels, dim=0)

# Mimic HF EvalPrediction for compatibility with your function
class EvalPrediction:
    def __init__(self, predictions, label_ids):
        self.predictions = predictions
        self.label_ids = label_ids

eval_pred = EvalPrediction(predictions=all_logits, label_ids=all_labels)

# Compute metrics
results = Eval.compute_metrics(eval_pred, predict_threshold=True)

print("Evaluation results:", results)
