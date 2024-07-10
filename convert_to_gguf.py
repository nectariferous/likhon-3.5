import argparse
import torch
from transformers import GPT2Tokenizer
from advanced_likhon_model import create_advanced_likhon35_model
from llama_cpp import Llama

def convert_to_gguf(input_model_path, output_model_path, quantization="q4_0"):
    # Load the original model
    model = create_advanced_likhon35_model()
    model.load_state_dict(torch.load(input_model_path))
    model.eval()

    # Convert model architecture to match Llama format
    llama_model = Llama.from_pretrained(model)

    # Save in GGUF format
    llama_model.save_pretrained(output_model_path, quantization=quantization)

    print(f"Model converted and saved to {output_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Likhon 3.5 model to GGUF format")
    parser.add_argument("--input_model", type=str, required=True, help="Path to the input Likhon 3.5 model")
    parser.add_argument("--output_model", type=str, required=True, help="Path to save the output GGUF model")
    parser.add_argument("--quantization", type=str, default="q4_0", choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"], help="Quantization level for GGUF model")
    
    args = parser.parse_args()
    
    convert_to_gguf(args.input_model, args.output_model, args.quantization)
