import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def fine_tune():
    model_name = './results/likhon-3.5-base'  # Path to the trained model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load the WikiText-2 dataset for fine-tuning
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']

    def tokenize_function(examples):
        return tokenizer(examples['text'], return_tensors="pt", padding='max_length', truncation=True, max_length=512)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_validation = validation_dataset.map(tokenize_function, batched=True)

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
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation
    )

    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained('./fine_tuned_results/likhon-3.5-finetuned')
    tokenizer.save_pretrained('./fine_tuned_results/likhon-3.5-finetuned')

if __name__ == "__main__":
    fine_tune()
