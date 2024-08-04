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
    tokenizer = PreTrainedTokenizer.from_pretrained(config.tokenizer_dir)
    # load config
    t5_config = get_T5_config(
        T5ModelConfig,
        vocab_size=len(tokenizer),
        decoder_start_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # initialize model
    model = TexttoTextModel(t5_config)
    
    # load dataset
    train_dataset = get_dataset(
        config.train_data_dir, split="train", tokenizer=tokenizer
    )

    # training args
    generation_config = GenerationConfig()
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.decoder_start_token_id = tokenizer.pad_token_id
    generation_config.max_new_tokens = 512
    generation_config.num_beams = 1
    generation_config.do_sample = False
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.keep_last_n_checkpoints,
        seed=config.seed,
        dataloader_buffer_size=config.dataloader_buffer_size,
        max_seq_length=config.max_seq_length,
        seed=config.seed,
        # what should be inside generation config?
        generation_config=generation_config,
        optim=config.optim,
    )
    
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, max_length = config.max_seq_length)
    empty_cuda_cache = MyTrainerCallback()
    
    # define the trainer
    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        tokenizer = tokenizer,
        data_collator = collator,
        callbacks = [empty_cuda_cache]
    )
    
    # train
    trainer.train()
    
    # save the model
    trainer.save_model(config.output_dir)

if __name__ == "__main__":
    config = Trainingconfig()
    pretrain(config)