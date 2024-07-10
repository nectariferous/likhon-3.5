import argparse
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import json

from advanced_likhon_model import create_advanced_likhon35_model

def evaluate_gpqa(model, tokenizer, device, dataset):
    model.eval()
    all_predictions = []
    all_labels = []

    for batch in tqdm(dataset, desc="Evaluating GPQA"):
        inputs = tokenizer(batch['question'], return_tensors='pt', padding=True, truncation=True).to(device)
        labels = torch.tensor(batch['answer']).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs[0]
            predictions = torch.argmax(logits, dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return {"accuracy": accuracy, "f1_score": f1}

def evaluate_mmlu(model, tokenizer, device, dataset):
    model.eval()
    all_predictions = []
    all_labels = []

    for batch in tqdm(dataset, desc="Evaluating MMLU"):
        inputs = tokenizer(batch['question'], return_tensors='pt', padding=True, truncation=True).to(device)
        labels = torch.tensor(batch['answer']).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs[0]
            predictions = torch.argmax(logits, dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return {"accuracy": accuracy, "f1_score": f1}

def evaluate_human_eval(model, tokenizer, device, dataset):
    model.eval()
    correct = 0
    total = 0

    for sample in tqdm(dataset, desc="Evaluating HumanEval"):
        prompt = sample['prompt']
        test_cases = sample['test_cases']

        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=1024, num_return_sequences=1, temperature=0.7)
        
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Execute generated code and test cases
        try:
            exec(generated_code)
            for test_case in test_cases:
                exec(test_case)
            correct += 1
        except Exception:
            pass
        total += 1

    pass_rate = correct / total
    return {"pass_rate": pass_rate}

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_advanced_likhon35_model(
        vocab_size=args.vocab_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head
    )
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    results = {}

    if "gpqa" in args.benchmarks:
        gpqa_dataset = load_dataset("EleutherAI/gpqa", split="test")
        results["GPQA"] = evaluate_gpqa(model, tokenizer, device, gpqa_dataset)

    if "mmlu" in args.benchmarks:
        mmlu_dataset = load_dataset("cais/mmlu", "all", split="test")
        results["MMLU"] = evaluate_mmlu(model, tokenizer, device, mmlu_dataset)

    if "humaneval" in args.benchmarks:
        humaneval_dataset = load_dataset("openai_humaneval", split="test")
        results["HumanEval"] = evaluate_human_eval(model, tokenizer, device, humaneval_dataset)

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("Evaluation results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Advanced Likhon 3.5 Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--benchmarks", nargs='+', default=["gpqa", "mmlu", "humaneval"], help="Benchmarks to evaluate on")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="File to save evaluation results")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size for the model")
    parser.add_argument("--n_embd", type=int, default=1024, help="Embedding dimension for the model")
    parser.add_argument("--n_layer", type=int, default=24, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=16, help="Number of attention heads")
    
    args = parser.parse_args()
    main(args)
