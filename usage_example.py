import torch
from transformers import GPT2Tokenizer
from advanced_likhon_model import create_advanced_likhon35_model

def load_model(model_path, device='cuda'):
    model = create_advanced_likhon35_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_length=max_length, 
            temperature=temperature, 
            top_p=top_p, 
            num_return_sequences=1,
            do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def answer_question(model, tokenizer, question):
    prompt = f"Q: {question}\nA:"
    return generate_text(model, tokenizer, prompt, max_length=150)

def write_code(model, tokenizer, task_description):
    prompt = f"Write Python code to {task_description}:\n\n```python\n"
    generated_code = generate_text(model, tokenizer, prompt, max_length=300, temperature=0.4)
    return generated_code.split("```")[1].strip() if "```" in generated_code else generated_code

def explain_concept(model, tokenizer, concept):
    prompt = f"Explain the concept of {concept} in simple terms:\n\n"
    return generate_text(model, tokenizer, prompt, max_length=200)

if __name__ == "__main__":
    model_path = "path/to/your/likhon35_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(model_path, device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("Example 1: Generate Text")
    prompt = "The future of artificial intelligence is"
    generated_text = generate_text(model, tokenizer, prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}\n")

    print("Example 2: Answer Question")
    question = "What is the theory of relativity?"
    answer = answer_question(model, tokenizer, question)
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")

    print("Example 3: Write Code")
    task = "implement a binary search algorithm"
    code = write_code(model, tokenizer, task)
    print(f"Task: {task}")
    print(f"Generated code:\n{code}\n")

    print("Example 4: Explain Concept")
    concept = "quantum entanglement"
    explanation = explain_concept(model, tokenizer, concept)
    print(f"Concept: {concept}")
    print(f"Explanation: {explanation}")

