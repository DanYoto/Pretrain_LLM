from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    pre_tokenizers,
    Tokenizer
)
from transformers import PretrainedTokenizerFast

special_tokens = ["<s>", "</s>", "<unk>"]
vocab_size = 32000

# BPE
def training_tokenizer():
    tokenizer = Tokenizer(models.BPE())
    
    # specify noirmalizers and pre-tokenizers
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.StripAccents()
    ])
    
    # specify pre-tokenizers
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=True)
    ])
    tokenizer.add_special_tokens(special_tokens)    
    
    trainer = trainers.BpeTrainer(vocab_size = vocab_size, min_frequency = 50, show_progress = True, special_tokens = special_tokens)
    tokenizer.train_from_iterator(trainer, ["data/train.txt"])
    tokenizer.save("tokenizer.json")
    
    # save tokenizer to PretrainedTokenizerFast to use huggingface's pipeline
    tokenizer = PretrainedTokenizerFast(tokenizer_object = tokenizer,
                                        bos_token = "<s>",
                                        eos_token = "</s>",
                                        unk_token = "<unk>")
    tokenizer.save_pretrained("tokenizer")