import sys

import torch
import transformers

import Model_Loader


from huggingface_hub import login
import os

token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not token:
    raise ValueError("HUGGINGFACE_HUB_TOKEN env variable not set!")

login(token=token)


def setup_pipeline():
    model, tokenizer = Model_Loader.load_model()
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline, tokenizer


def get_system_prompt():
    return {
        "role": "system",
        "content": (
            "You are an expert inorganic chemist. Determine if the following compound is "
            "likely to be synthesizable based on its composition, answering only “P” "
            "(for positive or possible) and “U” (for unknown or unlikely)."
        )
    }


def generate_response(pipeline, messages, eos_token):
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=eos_token,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    try:
        return outputs[0]["generated_text"][-1]["content"]
    except (KeyError, IndexError, TypeError):
        return outputs[0]["generated_text"]


def chat_loop(pipeline, tokenizer):
    messages = [get_system_prompt()]
    eos_token = tokenizer.eos_token_id

    print("Enter a compound composition (or type 'exit' or 'quit' to close the program):\n")
    sys.stdout.flush()
    user_input = "Ba8Ga16Sn30"
    messages.append({"role": "user", "content": user_input})
    response = generate_response(pipeline, messages, eos_token)
    print(f"\nModel output:\n{response}\n")
    messages.append({"role": "assistant", "content": response})


#if __name__ == "__main__":
#    try:
#        pipeline, tokenizer = setup_pipeline()
#        chat_loop(pipeline, tokenizer)
#    except KeyboardInterrupt:
#        print("\nInterrupted. Exiting.")
