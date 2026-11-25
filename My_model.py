import torch,os
import torch.nn as nn

# 创建一个简单的卷积神经网络
class My_model(nn.Module):
    def __init__(self):
        super(My_model,self).__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(64*28*28,512)
        self.bn4 = nn.BatchNorm1d(512)
        # self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512,4)

    def forward(self,x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x,1)
        x = torch.relu(self.bn4(self.fc1(x)))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    from dataloader import train_loader,val_loader
    from train import train_model,evaluate_model
    torch.manual_seed(42)    # 设置随机种子,保证两次实验初始化权重完全一样

    model = My_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    train_model(model,train_loader,criterion,optimizer,val_loader=val_loader,num_epochs=10)

    # 保存模型
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),'checkpoints','My_model','best.pth')
    best_model = My_model().to(device)
    best_model.load_state_dict(torch.load(ckpt,map_location=device))
    val_acc,val_loss = evaluate_model(best_model,val_loader)
    print(f"val_acc={val_acc:.4f}  val_loss={val_loss:.4f}")
