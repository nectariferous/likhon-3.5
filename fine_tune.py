import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer

def fine_tune():
    model_name = 'path_to_pretrained_model'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load your fine-tuning dataset here
    # fine_tune_dataset = ...

    training_args = TrainingArguments(
        output_dir='./fine_tuned_results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=fine_tune_dataset['train'],
        eval_dataset=fine_tune_dataset['validation']
    )

    trainer.train()

if __name__ == "__main__":
    fine_tune()
