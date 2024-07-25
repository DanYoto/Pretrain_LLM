from transformers import T5ForConditionalGeneration, T5Config
from transformers import (
    TextIteratorStreamer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from transformers.generation.configuration_utils import GenerationConfig
import torch


class TexttoTextModel(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)

    @torch.no_grad()
    def generate_text(
        self,
        input_ids,
        attention_mask,
        max_seq_len=128,
        search_type="greedy",
        streamer: TextIteratorStreamer = None,
    ):
        generation_config = GenerationConfig()
        # Differentiate eos and pad tokens
        generation_config.eos_token_id = 1
        generation_config.pad_token_id = 0
        generation_config.max_new_tokens = max_seq_len

        if search_type == "greedy":
            generation_config.num_beams = 1
            generation_config.do_sample = False
        elif search_type == "beam":
            generation_config.num_beams = 4
            generation_config.top_k = 50
            generation_config.do_sample = True
            generation_config.top_p = 0.95
            # to avoid gradients explosion, stop early if patient has been reached
            generation_config.early_stopping = True
        elif search_type == "sampling":
            generation_config.num_beams = 1
            generation_config.do_sample = True
            generation_config.top_k = 50
            generation_config.top_p = 0.95
            generation_config.temperature = 0.7

        return self.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            streamer=streamer,
        )


class MyTrainerCallback(TrainerCallback):
    log_cnt = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.log_cnt += 1
        if self.log_cnt % 5 == 0:
            print(f"Step {state.global_step} | Loss: {state.loss}")
            torch.cuda.empty_cache()

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        control.should_save = True
        return control
