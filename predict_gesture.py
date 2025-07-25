import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from model import GestureRecognitionModel   # Ensure this matches your model file

# ----- CONFIG -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_folder = "test_clip"  # Folder with 20 frames
model_path = "gesture_model.pth"
gesture_labels = ['Fist', 'Four', 'Me', 'One', 'Small']  # same order used during training
num_classes = len(gesture_labels)

# ----- TRANSFORM -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ----- LOAD MODEL -----
model = GestureRecognitionModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ----- LOAD FRAMES -----
frames = sorted(os.listdir(clip_folder))
frame_tensors = []

for frame_file in frames:
    img_path = os.path.join(clip_folder, frame_file)
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image)
    frame_tensors.append(tensor)

# Stack and add batch dimension
clip_tensor = torch.stack(frame_tensors).unsqueeze(0).to(device)  # shape: [1, 20, 3, 224, 224]

# ----- PREDICT -----
with torch.no_grad():
    output = model(clip_tensor)  # shape: [1, num_classes]
    predicted_idx = output.argmax(dim=1).item()
    predicted_label = gesture_labels[predicted_idx]

print(f"\n Predicted Gesture: {predicted_label}")