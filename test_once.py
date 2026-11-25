import os,torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
from dataloader import test_loader
from train import evaluate_model, device

if __name__ == '__main__':
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'resnet', 'best.pth')
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device)

    test_acc, test_loss = evaluate_model(model, test_loader)
    print(f"===== test 集一次性评估 =====")
    print(f"test_acc={test_acc:.4f}  test_loss={test_loss:.4f}  "
          f"test_p={np.exp(-test_loss):.4f}")