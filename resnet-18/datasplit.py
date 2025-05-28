from sklearn.model_selection import train_test_split
import os
import shutil

# 输入和输出路径
data_dir = "resnet-18/caltech-101/101_ObjectCategories"
output_dir = "resnet-18/caltech-101/split_data"  # 保存划分结果的根目录

# 创建输出文件夹（如果不存在）
os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

# 过滤隐藏文件（如.DS_Store）
classes = [f for f in os.listdir(data_dir) if not f.startswith('.')]
print(f"Total classes: {len(classes)}")

# 划分比例
train_ratio, val_ratio = 0.7, 0.15  # 测试集自动为 1 - 0.7 - 0.15 = 0.15

# 随机种子（确保可复现）
random_seed = 42

for cls in classes:
    # 原始图像路径
    cls_dir = os.path.join(data_dir, cls)
    images = [f for f in os.listdir(cls_dir) if not f.startswith('.')]
    
    # 划分数据集
    train, test = train_test_split(images, train_size=train_ratio, random_state=random_seed)
    val, test = train_test_split(test, train_size=val_ratio/(1-train_ratio), random_state=random_seed)
    
    # 创建类别子目录（train/val/test）
    os.makedirs(os.path.join(output_dir, "train", cls), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", cls), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", cls), exist_ok=True)
    
    # 复制文件到目标文件夹
    for img in train:
        src = os.path.join(cls_dir, img)
        dst = os.path.join(output_dir, "train", cls, img)
        shutil.copy2(src, dst)  # 保留文件元数据
    
    for img in val:
        src = os.path.join(cls_dir, img)
        dst = os.path.join(output_dir, "val", cls, img)
        shutil.copy2(src, dst)
    
    for img in test:
        src = os.path.join(cls_dir, img)
        dst = os.path.join(output_dir, "test", cls, img)
        shutil.copy2(src, dst)

print("Dataset splitting completed!")