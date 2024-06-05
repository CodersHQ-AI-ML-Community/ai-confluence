import torch
from PIL import Image
import torchvision.transforms as transforms

from utils import DEVICE, NORMALIZE, PIXELS

def predict_image(model, image_path, class_names, device=DEVICE):
    # Load your JPEG image
    image = Image.open(image_path)
    mean, std = NORMALIZE
    # Define the transformation: resize to 32x32 and convert to a tensor
    transform = transforms.Compose(
        [
            transforms.Resize(size=PIXELS),  # Resize the image to 32x32 pixels
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Apply the transformation
    cifar_image = transform(image).to(device)

    # The cifar_image tensor will have the shape [3, 32, 32]
    # corresponding to the [channels, height, width] expected by CIFAR models

    print("Transformed image shape:", cifar_image.shape)

    input_image = cifar_image.unsqueeze(0)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(input_image)

    # No need for torch.max if output is directly the predicted class in a single output scenario.
    # This depends on how your network architecture is set up and the final layer.
    predicted = torch.max(outputs, 1)[1]  # Gets the index of the max log-probability

    # Since we are only predicting one image, we don't need a loop to print the prediction
    predicted_class = class_names[predicted.item()]
    print("Predicted:", predicted_class)

    return predicted_class
