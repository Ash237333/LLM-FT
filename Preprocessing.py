import json
import os

import torch
from datasets import Dataset, DatasetDict
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizer

token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not token:
    raise ValueError("HUGGINGFACE_HUB_TOKEN env variable not set!")
login(token=token)

SYSTEM_PROMPT = ("You are an expert inorganic chemist. Determine if the following compound is "
                 "likely to be synthesizable based on its composition, answering only"
                 " 'P' (for positive or possible) and 'U' (for unknown or unlikely).")


def load_model():
    """
    Downloads, caches, and loads the model and tokenizer with 8-bit quantization.

    Returns:
        tuple:
            Model (AutoModelForCausalLM): The quantized language model.
            Tokenizer (AutoTokenizer): The corresponding tokenizer.
    """

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )

    return model, tokenizer


def load_custom_dataset(dataset_path, test_split):
    """
       Loads a custom dataset from a JSON file and splits it into train and test sets.

       The input JSON file should contain two keys: "positive" and "unlabeled", each containing a list of compounds.
       The "positive" samples are labeled as "P", and the "unlabeled" samples are labeled as "U".
       The data is split into train and test sets while preserving the label distribution using stratified sampling.

       Args:
           dataset_path (str): Path to the input JSON file containing "positive" and "unlabeled" entries.
           test_split (float): Fraction of the dataset to reserve for testing (e.g., 0.2 for 20%).

       Returns:
           DatasetDict: A Hugging Face DatasetDict with 'train' and 'test' splits, each containing examples
               with 'instruction' and 'response' fields.
       """
    with open(dataset_path) as f:
        raw_data = json.load(f)

    data = [{"instruction": t, "response": "P"} for t in raw_data.get("positive", [])] + \
           [{"instruction": t, "response": "U"} for t in raw_data.get("unlabeled", [])]
    dataset = Dataset.from_list(data)
    dataset = dataset.train_test_split(test_size=test_split, stratify_by_column="response")
    return DatasetDict(dataset)


def prep_dataset(dataset, tokenizer):
    """
    Applies tokenization and formatting to a dataset using a chat-style template.

    Each example is passed through the `tokenize_chat` function, which formats the input as a conversation
    and tokenizes it for language model training. The original 'instruction' and 'response' fields are removed
    after tokenization.

    Args:
        dataset (DatasetDict): A Hugging Face DatasetDict with 'train' and 'test' splits. Each example must contain
            'instruction' and 'response' fields.
        tokenizer (PreTrainedTokenizer): A tokenizer with a chat template (e.g., from Hugging Face Transformers)
            that supports `apply_chat_template`.

    Returns:
        DatasetDict: A new DatasetDict where each example has 'input_ids', 'attention_mask', and 'labels' fields,
        ready for training.
    """
    return dataset.map(
        lambda x: tokenize_chat(x, tokenizer),
        batched=False,
        remove_columns=["instruction", "response"]
    )


def tokenize_chat(example, tokenizer):
    """
      Tokenizes a single example as a chat conversation and prepares training labels for causal language modeling.

      Constructs a conversation using the system prompt, user instruction, and assistant response.
      Tokenizes the full chat, then masks the prompt tokens in the label sequence with -100 so that only the
      assistant response contributes to the loss.

      Args:
          example (dict): A dictionary with 'instruction' and 'response' fields representing the user input
              and assistant output.
          tokenizer (PreTrainedTokenizer): A tokenizer that supports chat formatting via `apply_chat_template`.

      Returns:
          dict: A dictionary with 'input_ids', 'attention_mask', and 'labels' suitable for language model training.
      """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]}
    ]

    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokenized = tokenizer(full_text, truncation=False, padding=False)

    labels = tokenized["input_ids"].copy()

    response_start = full_text.rfind(example["response"])
    prompt_part = full_text[:response_start]

    prompt_token_len = len(tokenizer(prompt_part)["input_ids"])

    labels[:prompt_token_len] = [-100] * prompt_token_len

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }


class DataCollator:
    def __init__(self, tokenizer):
        """
        Custom data collator to pad the chat versions of input_ids, attention_masks and labels

        Pads input sequences and attention masks to the same length using the tokenizer's padding method.
        Also, manually pads the label sequences with -100 to ignore the loss on padding tokens.

        Attributes:
            tokenizer (PreTrainedTokenizer): A tokenizer instance from Hugging Face Transformers.
        """
        self.tokenizer = tokenizer

    def __call__(self, batch):
        """
        Prepares a batch of tokenized examples by padding input IDs, attention masks, and labels.

        Args:
            batch (list[dict]): A list of examples, where each example is a dictionary containing
                'input_ids', 'attention_mask', and 'labels'.

        Returns:
            dict: A dictionary with keys 'input_ids', 'attention_mask', and 'labels', each as a
            padded PyTorch tensor suitable for model input.
        """
        input_ids = [example["input_ids"] for example in batch]
        attention_mask = [example["attention_mask"] for example in batch]
        labels = [example["labels"] for example in batch]

        batch_encoding = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt"
        )

        max_len = batch_encoding["input_ids"].shape[1]
        padded_labels = [label + [-100] * (max_len - len(label)) for label in labels]
        batch_encoding["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch_encoding
