import torch
from train.train_without_pretrain import build_resnet18, create_dataloaders
from tqdm import tqdm

def chekpoint_path(pretrain=False):
    if pretrain:
        return "resnet-18/pretrain/best_resnet18_caltech101.pth"
    else:
        return "resnet-18/train/checkpoints/best_checkpoint.pth"  # 注意这里修改了文件名

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18().to(device)
    _, _, test_loader, _ = create_dataloaders()
    
    pretrain = False  # 切换为True来评估预训练模型
    
    checkpoint_path = chekpoint_path(pretrain=pretrain)
    checkpoint = torch.load(checkpoint_path)
    
    # 根据模型类型加载不同的状态字典
    if pretrain:
        model.load_state_dict(checkpoint)  # 预训练模型直接加载
    else:
        model.load_state_dict(checkpoint['model_state_dict'])  # 非预训练模型从字典中加载
    
    test_acc = evaluate(model, test_loader, device)
    print(f"Test Accuracy ({'Pretrained' if pretrain else 'From Scratch'}): {test_acc:.2f}%")