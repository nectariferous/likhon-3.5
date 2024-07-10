from model import Likhon35Model

model = Likhon35Model('path_to_fine_tuned_model')

response = model.generate("Explain the concept of quantum entanglement.")
print(response)
