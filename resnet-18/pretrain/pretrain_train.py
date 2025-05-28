#!/usr/bin/env python
# pretrain_train.py

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import create_dataloaders
from tqdm import tqdm

def main():
    # 1. 命令行参数
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-18 on Caltech-101")
    parser.add_argument('--batch-size',    type=int,   default=32,    help='batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num-epochs',    type=int,   default=20,    help='number of epochs')
    parser.add_argument('--log-dir',       type=str,   default='runs', help='TensorBoard log directory')
    parser.add_argument('--output-path',   type=str,   default='best_model.pth', help='where to save best model')
    parser.add_argument('--use-pretrained', action='store_true', help='use ImageNet pretrained weights')
    args = parser.parse_args()

    batch_size      = args.batch_size
    learning_rate   = args.learning_rate
    num_epochs      = args.num_epochs
    log_dir         = args.log_dir
    output_path     = args.output_path
    use_pretrained  = args.use_pretrained

    # 2. 设备 & 多卡检测
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device}, GPUs available: {num_gpus}")

    # 3. 构建模型
    model = models.resnet18(pretrained=use_pretrained)
    num_classes = 102  # Caltech-101有101类+1背景
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 4. 放到CUDA并包裹DataParallel（若有多卡）
    model = model.to(device)
    if device.type == 'cuda' and num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Wrapped model with DataParallel on {num_gpus} GPUs")

    # 5. 优化器：根据是否DataParallel来拿到正确的fc参数
    if isinstance(model, nn.DataParallel):
        if use_pretrained:
            params = model.module.fc.parameters()
        else:
            params = model.module.parameters()
    else:
        if use_pretrained:
            params = model.fc.parameters()
        else:
            params = model.parameters()
    optimizer = optim.Adam(params, lr=learning_rate)

    # 6. 损失函数
    criterion = nn.CrossEntropyLoss()

    # 7. TensorBoard 日志
    writer = SummaryWriter(log_dir)

    # 8. 数据加载
    train_loader, val_loader, test_loader = create_dataloaders(batch_size)

    # 9. 训练与验证循环
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{num_epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Train/Loss', avg_loss, epoch)

        # 验证
        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        # 记录到TensorBoard
        writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        writer.add_scalar('Val/Accuracy', val_acc, epoch)
        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            state_dict = (model.module.state_dict() if isinstance(model, nn.DataParallel)
                          else model.state_dict())
            torch.save(state_dict, output_path)
            print(f"  → Saved best model to {output_path}")

    # 10. 测试集评估
    checkpoint = torch.load(output_path, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_acc = 100. * correct / total
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"TensorBoard logs in {log_dir}, best model saved to {output_path}")

if __name__ == '__main__':
    main()

# 运行完毕后，可以使用以下命令启动TensorBoard：
# tensorboard --logdir=/work/home/maben/project/homework/neural_network_deep_learning/mid_hw/resnet-18/pretrain/logs --port=8080