from transformers import (
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
)
from transformers.generation.configuration_utils import GenerationConfig

from ..utils import *
from ..customized_model.cus_model import *
from ..config import *


def pretrain(config: Trainingconfig):
    # load tokenizer
    tokanizer = PreTrainedTokenizer.from_pretrained(config.tokenizer_dir)
    # load config
    t5_config = get_T5_config(
        T5ModelConfig,
        vocab_size=len(tokanizer),
        decoder_start_token_id=tokanizer.pad_token_id,
        eos_token_id=tokanizer.eos_token_id,
    )
    # initialize model
    model = CustomizedT5(t5_config)
    # load dataset
    train_dataset = get_dataset(
        config.train_data_dir, split="train", tokenizer=tokanizer
    )

    # unfinished
