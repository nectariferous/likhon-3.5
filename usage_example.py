from model import Likhon35Model

model = Likhon35Model('./fine_tuned_results/likhon-3.5-finetuned')  # Path to the fine-tuned model

response = model.generate("Explain the concept of quantum entanglement.")
print(response)
