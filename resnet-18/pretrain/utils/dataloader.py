import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

# 数据路径（根据你的实际路径修改）
data_dir = "/work/home/maben/project/homework/neural_network_deep_learning/mid_hw/resnet-18/caltech-101/split_data"  # 包含 train/val/test 子目录

# 定义数据增强和归一化
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),      # 随机裁剪缩放
    transforms.RandomHorizontalFlip(),       # 水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 统计量
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),                  # 调整大小
    transforms.CenterCrop(224),              # 中心裁剪
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
def load_datasets():
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=val_test_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform=val_test_transform
    )
    return train_dataset, val_dataset, test_dataset

# 创建 DataLoader
def create_dataloaders(batch_size=32):
    train_dataset, val_dataset, test_dataset = load_datasets()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader

# 测试代码
if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    print(f"Class names: {train_loader.dataset.classes}")  # 输出类别名称