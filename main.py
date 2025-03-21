import torch
import requests
from PIL import Image
from torchvision import transforms
import gradio as gr

model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True).eval()

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

# Defining a predict function
def predict(inp):
    """
    The function converts the input image into a PIL Image and subsequently into a PyTorch tensor.
    After processing the tensor through the model, it returns the predictions in the form of a dictionary named confidences.
    """
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
    return confidences


# Interface
gr.Interface(fn=predict,
       inputs=gr.Image(type="pil"),
       outputs=gr.Label(num_top_classes=3),
       examples=["./content/lion.jpeg", "./content/cheetah.jpeg"]).launch(share=True)