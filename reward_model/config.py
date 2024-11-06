from types import SimpleNamespace
from transformers import TrainingArguments, BitsAndBytesConfig
import torch
from peft import LoraConfig

WANDB_PROJECT = "RL_experiment"
WANDB_ENTITY = "digital-ethics-responsible-ai"
WANDB_TAGS = [
    #"meta-llama/Llama-3.2-1B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Reinforcement_Learning_experiment",
    "reward_modeling",
    "rotera",
    #"agentic methods for function calling",
]

config = SimpleNamespace(
    dataset_path="trl-lib/ultrafeedback_binarized",
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    per_device_train_batch_size=3,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=1e-3,
    gradient_checkpointing=True,
    max_seq_length=2048,
    freeze_embed=False,
    bf16=True,
    lr_scheduler_type="cosine",
    use_lora=False,
    max_steps=-1,
    num_train_epochs=3,
)
   
training_args = TrainingArguments(
    output_dir="/home/azureuser/cloudfiles/code/Users/yutong.jiang2/autolayout/sft_and_dpo_script/sft_scripts/output",
    per_device_train_batch_size=config.per_device_train_batch_size,
    bf16=config.bf16,
    learning_rate=config.learning_rate,
    lr_scheduler_type=config.lr_scheduler_type,
    warmup_ratio=0.1,
    max_steps=config.max_steps,
    num_train_epochs=config.num_train_epochs,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    gradient_checkpointing=config.gradient_checkpointing,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    evaluation_strategy="no",
    # logging strategies
    report_to="wandb",
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="no",
    remove_unused_columns=False
)

model_kwargs = dict(
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    use_cache=False,
    device_map="auto",
    # device_map = None,
    # quantization_config = quantization_config,
)

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

peft_config = LoraConfig(
    r=32,  # the rank of the LoRA matrices
    lora_alpha=16,  # the weight
    lora_dropout=0.1,  # dropout to add to the LoRA layers
    bias="none",  # add bias to the nn.Linear layers
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],  # the name of the layers to add LoRA
)