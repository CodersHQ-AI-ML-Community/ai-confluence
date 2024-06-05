import torch
import pickle

class ModelSaveLoad:
    def save_model(self, model, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        model.eval()
        print(f"Model loaded from {filepath}")
        return model