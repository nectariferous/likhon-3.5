import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate():
    model_name = 'path_to_fine_tuned_model'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load your evaluation dataset here
    # eval_dataset = ...

    for example in eval_dataset:
        inputs = tokenizer(example['input'], return_tensors="pt")
        outputs = model.generate(**inputs)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    evaluate()
