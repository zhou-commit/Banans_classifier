'''===加载数据集==='''
import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 显式指定数据集根目录，
root_dir = "./bananas_dataset/classifier"
train_path = os.path.join(root_dir,"train")
test_path = os.path.join(root_dir,"test")
val_path = os.path.join(root_dir,"val")

# 创建争对训练集的数据预处理,不同于测试集和验证集
def build_transforms(is_train):
    if is_train:
        return transforms.Compose([
        transforms.RandomResizedCrop(288,scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),                
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    else:
        return transforms.Compose([
        transforms.Resize((288,288)),
        transforms.CenterCrop(288),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

# 创建一个工厂函数
def build_dataset(root,is_train):
    return ImageFolder(root, transform=build_transforms(is_train))

def build_loader(root,is_train):
    ds = build_dataset(root,is_train)
    return DataLoader(
        ds,
        batch_size=32,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=1,
    )

# 获取单张图片的完整路径
img_path = build_dataset(train_path,is_train=True).samples[0][0]
img_label = build_dataset(train_path,is_train=True).samples[0][1]
'''samples[idx]->[完整路径，索引类别]，代表一个元组，train_dataset是一个list[],list里面是元组
samples[idx][0]代表元组的第一个元素，即完整路径，samples[idx][1]代表元组的第二个元素，即索引类别'''

# 创建数据集
train_dataset = build_dataset(train_path,is_train=True)
val_dataset = build_dataset(val_path,is_train=False)
test_dataset = build_dataset(test_path,is_train=False)
# print(train_dataset.classes, train_dataset.class_to_idx)

# 加载数据集   
train_loader = build_loader(train_path,is_train=True)
val_loader = build_loader(val_path,is_train=False)
test_loader = build_loader(test_path,is_train=False)

if __name__ == "__main__":
    for img,labels in train_dataset:
        print(img.shape)


