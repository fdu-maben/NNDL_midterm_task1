import argparse
import torch
import torch.nn as nn
from torchvision import models
from utils.dataloader import create_dataloaders

def main():
    # 1. Command line argument
    parser = argparse.ArgumentParser(description="Evaluate a trained ResNet-18 model")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    args = parser.parse_args()

    ckpt_path  = args.ckpt_path
    batch_size = args.batch_size

    # 2. Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device}, GPUs available: {num_gpus}")

    # 3. Load model
    model = models.resnet18(pretrained=False)
    num_classes = 102  # As in pretrain_train.py
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 4. Wrap with DataParallel if multiple GPUs
    model = model.to(device)
    if device.type == 'cuda' and num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Wrapped model with DataParallel on {num_gpus} GPUs")

    # 5. Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {ckpt_path}")

    # 6. Load data
    _, _, test_loader = create_dataloaders(batch_size)

    # 7. Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_acc = 100. * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == '__main__':
    main()
