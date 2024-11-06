from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from config import *
from datasets import load_dataset
import torch.nn as nn
import wandb

def build_dataset(tokenizer, train_data_path):
    def tokenize(sample):
        sample['positive'] = tokenizer.apply_chat_template(
            sample['chosen'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        sample['negative'] = tokenizer.apply_chat_template(
            sample['rejected'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        
        tokenized_pos = tokenizer(sample['positive'], truncation=True)
        tokenized_neg = tokenizer(sample['negative'], truncation=True)
        sample["input_ids_acc"] = tokenized_pos["input_ids"]
        sample["attention_mask_acc"] = tokenized_pos["attention_mask"]
        sample["input_ids_rej"] = tokenized_neg["input_ids"]
        sample["attention_mask_rej"] = tokenized_neg["attention_mask"]
        return sample
    
    train_data = load_dataset(train_data_path, split="train").shuffle(seed=42)
    train_data = train_data.map(tokenize, num_proc=8)
    return train_data


# data collator should be rewritten as the input data contains both acc and rej
# pad input batch to the longest sequence
class RewardDataCollatorWithPadding:
    def __init__(self, tokenizer, padding, max_length, pad_to_multiple_of, return_tensors):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, data_samples):
        merged_features = []
        for data_sample in data_samples:
            merged_features.append({"input_ids_acc": data_sample['input_ids_acc'], "attention_mask_acc": data_sample['attention_mask_acc']})
            merged_features.append({"input_ids_rej": data_sample['input_ids_rej'], "attention_mask_rej": data_sample['input_ids_rej']})

        batch = self.tokenizer.pad(merged_features, padding = self.padding, max_length = self.max_length, return_tensors = self.return_tensors)
        return {
            "input_ids": batch['input_ids'],
            "attention_mask": batch['attention_mask'],
            "return_loss": True
        }

class RewardTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://huggingface.co/papers/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

if __name__ == "__main__":
        
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        job_type="train",
        tags=WANDB_TAGS,
        group="room_layout",
        config=config,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.truncation_side = "left"
    # in case tokenizer doesn't have pad token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = config.max_seq_length
    train_dataset = build_dataset(tokenizer, config.dataset_path)

    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, torch_dtype = torch.bfloat16, device_map = "auto", trust_remote_code = True, num_labels = 1)

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()
    save_path = f"{training_args.output_dir}/reinforcement_learning_trial_1"
    trainer.save_model(save_path)


    model_at = wandb.Artifact(
        name=f"mistral_7b_inst_v2___{wandb.run.id}",
        type="model",
        description="mistral_7b_inst_v2 model fine-tuned for rotera desk case",
        metadata={"finetuned_from": config.model_id},
    )
    model_at.add_dir(save_path)
    wandb.log_artifact(model_at)