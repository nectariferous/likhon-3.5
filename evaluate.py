import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def evaluate():
    model_name = './fine_tuned_results/likhon-3.5-finetuned'  # Path to the fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load the WikiText-2 dataset for evaluation
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    validation_dataset = dataset['validation']

    def tokenize_function(examples):
        return tokenizer(examples['text'], return_tensors="pt", padding='max_length', truncation=True, max_length=512)

    tokenized_validation = validation_dataset.map(tokenize_function, batched=True)

    for example in tokenized_validation:
        inputs = {key: torch.tensor(val) for key, val in example.items() if key in tokenizer.model_input_names}
        outputs = model.generate(**inputs)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    evaluate()
