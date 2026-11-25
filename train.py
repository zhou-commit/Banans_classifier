import torch,random,os
from dataloader import train_loader,test_loader,val_loader,train_dataset,val_dataset
from torch.utils.data import Subset
import torchvision.models as models   # 构建resnet18模型models.resnet18
import numpy as np
from tqdm import tqdm
import torch.nn as nn

# 设计随机种子实现代码复现
def set_seed(seeds):
    torch.manual_seed(seeds)
    torch.cuda.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seeds)
    random.seed(seeds)
set_seed(0)

# 设置设备使用为GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练模型的训练函数
def train_model(model,dataloader,criterion,optimizer,val_loader=None,num_epochs=20,scheduler=None):
    ''' 训练resnet18模型并保存 '''
    # ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'checkpoints')
    ''' 训不同的模型并保存，目的是为了隔离不同的模型文件 '''
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'checkpoints',type(model).__name__.lower())
    os.makedirs(ckpt_dir,exist_ok=True)
    best_loss = float('inf')
    print(f"lr={optimizer.param_groups[0]['lr']}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0.0
        total = 0.0

        with tqdm(dataloader,unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs} [train]")
            for i,(inputs,labels) in enumerate(tepoch):
                inputs,labels = inputs.to(device),labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total += labels.size(0)
                _,predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                accuracy = 100. * correct / total
                tepoch.set_postfix(loss=running_loss/(i+1),accuracy=accuracy)
                tepoch.update()
            
# 每个epoch结束后需要在验证集上进行评估，并且评估是否保存模型
        if val_loader is not None:
            val_accuracy,val_loss = evaluate_model(model,val_loader)
            print(f"Epoch {epoch+1}/{num_epochs} train_acc={accuracy:.4f}% train_loss={running_loss/(i+1):.4f} "
                f"val_acc={val_accuracy:.4f}% val_loss={val_loss:.4f} "
                f"val_p={np.exp(-val_loss):.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(),os.path.join(ckpt_dir,'best.pth'))
            else:
                print(f"Epoch {epoch+1}/{num_epochs} train_acc={accuracy:.4f} train_loss={running_loss/(i+1):.4f}")
        torch.save(model.state_dict(),os.path.join(ckpt_dir,f'cp{epoch}.pth')) 

        if scheduler is not None:
            scheduler.step()

# 验证模型的验证函数，同时其也可以作为评估模型来使用
def evaluate_model(model,dataloader):      
    model.eval()
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for inputs,labels in dataloader:
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            running_loss += loss.item() * labels.size(0)
            _,predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            accuracy = 100. * correct / total
    return accuracy if total else 0.0,running_loss/total

# 构建模型
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
'''用残差连接的fc层替换最后一层的全连接层'''
resnet18.fc = torch.nn.Linear(resnet18.fc.in_features,4)

# 冻结骨干进行测试
FREEZE_FEATURES = False

if FREEZE_FEATURES:
    for param in resnet18.parameters():
        param.requires_grad = False
    for param in resnet18.fc.parameters():
        param.requires_grad = True

# 进行冒烟测试
SMOKE_TEST = False
SMOKE_N = 256

if SMOKE_TEST:
    n_train,n_val = len(train_dataset),len(val_dataset)
    n_idx1,n_idx2 = list(range(n_train)),list(range(n_val))
    random.shuffle(n_idx1)
    random.shuffle(n_idx2)

    idx1 = n_idx1[:SMOKE_N]
    idx2 = n_idx2[:SMOKE_N//2]
    train_data = Subset(train_dataset,idx1)
    val_data = Subset(val_dataset,idx2)
else:
    train_data = train_dataset
    val_data = val_dataset

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
if FREEZE_FEATURES:
        optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, resnet18.parameters()),lr=1e-4, weight_decay=1e-5
    )
else:
    optimizer = torch.optim.Adam([
        {'params': [p for n, p in resnet18.named_parameters() if not n.startswith('fc.')], 'lr': 1e-5},
        {'params': resnet18.fc.parameters(), 'lr': 1e-3},
])   # 采用差异化的一个学习率，越靠近输入层的采用小步长的lr进行微调，越靠近输出层的越与任务相关，分配大lr

if __name__ == '__main__':
    resnet18 = resnet18.to(device)
    epoch = 2 if SMOKE_TEST else 20

    # 加余弦退火+训15轮,scheduler是学习率调度器
    from torch.optim.lr_scheduler import CosineAnnealingLR  # scheduler必须绑定optimizer，因为它改的是optimizer的lr
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch)      # T_max = 总轮数
    train_model(resnet18,train_loader,criterion,optimizer,val_loader=val_loader,num_epochs=epoch,scheduler=scheduler)