import torch,os
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(3*224*224,128)  # 输入层
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128,64)         # 隐藏层
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64,4)           # 输出层

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

if __name__=='__main__':
    from train import train_model,evaluate_model  # 在守卫里面导入就不会导致循环导入
    from dataloader import train_loader,val_loader
    torch.manual_seed(42)

    model = MLP().to(device)
    # 自己定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    train_model(model,train_loader,criterion,optimizer,val_loader=val_loader,num_epochs=10) 

    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'mlp','best.pth')
    best_model = MLP().to(device)
    best_model.load_state_dict(torch.load(ckpt, map_location=device))
    val_acc,val_loss = evaluate_model(best_model,val_loader)
    print(f"val_acc={val_acc:.4f}  val_loss={val_loss:.4f}")