import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = 'gpt-2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load your dataset here
    # dataset = ...

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation']
    )

    trainer.train()

if __name__ == "__main__":
    main()
