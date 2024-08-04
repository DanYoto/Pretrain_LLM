from dataclasses import dataclass
from transformers import T5Config

@dataclass
class Trainingconfig:
    epochs: int = 5
    per_device_train_batch_size: int = 8
    learning_rate: float = 1e-4
    div_factor: float = 1e4
    mix_precision: str = 'bf16'
    gradient_accumulation_steps: int = 8
    
    warmup_steps: int = 1024
    
    tokenizer_dir: str
    model_dir: str
    model_config_dir: str
    train_data_dir: str
    test_data_dir: str
    
    train_state_dir: str
    output_dir: str
    
    logging_steps: int = 100
    save_steps: int = 1000
    
    keep_last_n_checkpoints: int = 5
    
    seed: int = 42
    dataloader_buffer_size: int = 1024
    max_seq_length: int = 512
    
    optim = "adafaactor"
    
@dataclass
class T5ModelConfig:
    # dimension of fully connected layer
    d_ff: int = 2048
    # dimension of word embedding
    d_model: int = 512
    # number of heads
    num_heads: int = 8
    d_kv: int = 64
    
    num_decoder_layers: int = 10
    num_layers: int = 10