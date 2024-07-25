from datasets import load_dataset
import numpy as np
from config import *
from transformers import T5Config


def get_dataset(file, tokenizer):
    dataset = load_dataset("parquet", data_files=file)

    def token_to_ids(samples):
        eos_token_id = tokenizer.eos_token_id

        batch_prompt = samples["prompt"]
        batch_response = samples["response"]

        encoded_prompt = tokenizer(
            batch_prompt, truncation=False, padding=False, return_attention_mask=False
        )
        encoded_response = tokenizer(
            batch_response, truncation=False, padding=False, return_attention_mask=False
        )

        input_ids = [
            np.array(item + eos_token_id) for item in encoded_prompt["input_ids"]
        ]
        labels = [
            np.array(item + eos_token_id) for item in encoded_response["input_ids"]
        ]

        return {"input_ids": input_ids, "labels": labels}

    dataset = dataset.map(token_to_ids, batched=True, batch_size=128)
    return dataset


def get_T5_config(
    config: T5ModelConfig,
    vocab_size: int,
    decoder_start_token_id: int,
    eos_token_id: int,
):
    t5_config = T5Config(
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_heads=config.num_heads,
        d_kv=config.d_kv,
        num_decoder_layers=config.num_decoder_layers,
        num_layers=config.num_layers,
        vocab_size=vocab_size,
        decoder_start_token_id=decoder_start_token_id,
        eos_token_id=eos_token_id,
    )
    return t5_config
