from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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


def load_custom_dataset():
    """
    Loads and preprocesses a custom dataset from a text file.

    This function performs the following steps:
    1. Loads a text dataset from 'ICSD_compositions_valid.txt'.
    2. Renames the 'text' column to 'instruction'.
    3. Adds a new column 'response' with a constant value "P".
    4. Splits the dataset into training and test sets (80/20).

    Returns:
        DatasetDict: A dictionary with 'train' and 'test' splits, where each split is a Dataset object
            with columns:
            - instruction (str): The input text.
            - response (str): A constant response value ("P").
    """
    dataset = load_dataset("text", data_files="ICSD_compositions_valid.txt")
    dataset = dataset.rename_column("text", "instruction")

    # Add a response column with "P"
    def add_response(example):
        example["response"] = "P"
        return example

    dataset = dataset.map(add_response)
    dataset = dataset["train"].train_test_split(test_size=0.2)
    return dataset


def prep_dataset(dataset, tokenizer):
    return dataset.map(
        lambda x: tokenize_chat(x, tokenizer),
        batched=False,
        remove_columns=["instruction", "response"]
    )


def tokenize_chat(example, tokenizer):
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

    # Mask the prompt tokens so loss is only computed on the assistant response
    labels[:prompt_token_len] = [-100] * prompt_token_len

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }
