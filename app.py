import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image


class_names = [
    "Tomato Bacterial Spot", "Late Blight", "Early Blight",
    "Leaf Mold", "Septoria Leaf Spot", "Yellow Leaf Curl Virus",
    "Mosaic Virus", "Healthy"
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(class_names))
model.load_state_dict(torch.load("tomato_disease_model.pth", map_location=device))
model.to(device)
model.eval()


transform_fn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


medications = {
    "Tomato Bacterial Spot": "Use Copper-based fungicides and avoid overhead watering.",
    "Late Blight": "Apply fungicides like Chlorothalonil or Copper sprays.",
    "Early Blight": "Use Mancozeb or Copper-based fungicides. Rotate crops annually.",
    "Leaf Mold": "Improve ventilation and use fungicides like Chlorothalonil.",
    "Septoria Leaf Spot": "Apply fungicides with Chlorothalonil or Copper sulfate.",
    "Yellow Leaf Curl Virus": "Use insecticidal soaps and remove infected plants.",
    "Mosaic Virus": "No cure available. Remove infected plants and control aphids.",
    "Healthy": "No treatment needed. Maintain proper care and watering."
}

# Streamlit App UI
st.title(" Tomato Disease Classification App")
st.write("Upload an image of a tomato leaf to detect its disease and get treatment suggestions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)


    image_tensor = transform_fn(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()


    predicted_disease = class_names[class_idx]
    st.write(f"### Predicted Class: **{predicted_disease}** ")
    st.write(f"**Suggested Treatment:** {medications[predicted_disease]}")
