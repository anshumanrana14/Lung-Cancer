import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ========== CONFIG ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    "Benign cases",
    "Malignant cases",
    "Normal cases"
]

IMG_SIZE = 128  # must match training

# ========== MODEL ARCHITECTURE ==========
class LungCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(LungCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ========== LOAD MODEL ==========
model = LungCNN(num_classes=len(class_names))
model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()
print("✅ Model loaded successfully!")

# ========== TRANSFORMS ==========
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ========== INFERENCE FUNCTION ==========
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_class = class_names[predicted.item()]
    confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()

    # Show image
    plt.imshow(image)
    plt.title(f"Prediction: {predicted_class} ({confidence:.2f})")
    plt.axis("off")
    plt.show()

    return predicted_class, confidence

# ========== RUN ==========
image_path = "test.jpg"  # 👈 change this
pred_class, conf = predict_image(image_path)
print(f"\n🧠 Predicted: {pred_class}")
print(f"📊 Confidence: {conf:.4f}")
