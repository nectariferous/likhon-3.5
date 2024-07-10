import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def main():
    model_name = 'gpt-2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load the WikiText-2 dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']

    def tokenize_function(examples):
        return tokenizer(examples['text'], return_tensors="pt", padding='max_length', truncation=True, max_length=512)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_validation = validation_dataset.map(tokenize_function, batched=True)

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
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation
    )

    trainer.train()

    # Save the trained model
    model.save_pretrained('./results/likhon-3.5-base')
    tokenizer.save_pretrained('./results/likhon-3.5-base')

if __name__ == "__main__":
    main()
