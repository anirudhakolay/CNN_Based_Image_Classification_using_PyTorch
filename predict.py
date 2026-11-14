import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Import the model architecture from train.py
from train import SimpleCNN


def predict_image(image_path, model_path='best_model.pth'):
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint '{model_path}' not found! Please run train.py first.")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found!")
        return

    # Select Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get('classes', ['cat', 'dog', 'wild'])

    model = SimpleCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Preprocessing (same resolution and normalization as validation)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    top_prob, top_idx = torch.topk(probabilities, 1)
    predicted_class = class_names[top_idx.item()]
    confidence = top_prob.item() * 100.0

    print("=" * 45)
    print(f"Image: {image_path}")
    print(f"Prediction: {predicted_class.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 45)
    print("Class Probabilities:")
    for idx, class_name in enumerate(class_names):
        print(f"  - {class_name:5s}: {probabilities[idx].item() * 100:.2f}%")
    print("=" * 45)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image> [path_to_model]")
        print("Example: python predict.py val/cat/flickr_cat_000001.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else 'best_model.pth'
    predict_image(img_path, ckpt_path)

