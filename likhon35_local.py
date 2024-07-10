import argparse
from llama_cpp import Llama

def generate_text(model, prompt, max_tokens=100, temperature=0.7, top_p=0.95):
    output = model(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    return output['choices'][0]['text']

def main(args):
    # Load the GGUF model
    model = Llama(model_path=args.model_path, n_ctx=2048, n_gpu_layers=-1)  # Use all available GPU layers

    while True:
        prompt = input("Enter your prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break

        response = generate_text(model, prompt, args.max_tokens, args.temperature, args.top_p)
        print(f"Likhon 3.5: {response}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Likhon 3.5 GGUF model locally")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the GGUF model")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    
    args = parser.parse_args()
    main(args)

