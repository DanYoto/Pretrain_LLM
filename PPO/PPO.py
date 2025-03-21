from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer
from config import *
from datasets import load_dataset
import wandb

def build_dataset(dataset, tokenizer):
    def tokenize(element):
        input_ids = tokenizer.apply_chat_template(element['messages'][:1], padding = False, add_generation_prompt = True)
        return {"input_ids": input_ids, "lengths": len(input_ids)}
    return dataset.map(tokenize, remove_columns = dataset.column_names, num_proc = 4)
     

if __name__ == "__main__":

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        job_type="train",
        tags=WANDB_TAGS,
        group="rl_test",
        config=config,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, padding_side = "left")
    tokenizer.truncation_side = "left"
    # in case tokenizer doesn't have pad token
    tokenizer.pad_token = tokenizer.eos_token

    value_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_name, num_labels = 1)
    reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_name, num_labels = 1)

    ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_name)
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_name)

    train_data = load_dataset(config.dataset_path)['train']
    eval_data = load_dataset(config.dataset_path)['validation'] if training_args.eval_strategy != "no" else None

    train_dataset = build_dataset(train_data, tokenizer)
    if eval_data is not None:
        eval_dataset = build_dataset(eval_data, tokenizer)
        eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=4)
        
    train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=4)
    

    trainer = PPOTrainer(
        config = training_args,
        processing_class = tokenizer,
        policy = policy,
        ref_policy = ref_policy,
        reward_model = reward_model,
        value_model = value_model,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset
    )

    trainer.train()
    save_path = f"{training_args.output_dir}/mistral_7b_inst_v2_rotera_desk_2nd_trial"
    trainer.save_model(save_path)

    model_at = wandb.Artifact(
        name=f"mistral_7b_inst_v2___{wandb.run.id}",
        type="model",
        description="mistral_7b_inst_v2 model fine-tuned for rotera desk case",
        metadata={"finetuned_from": config.model_id},
    )
    model_at.add_dir(save_path)

    # log the model into the artifact
    wandb.log_artifact(model_at)
