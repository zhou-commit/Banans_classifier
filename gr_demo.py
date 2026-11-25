import gradio as gr
import torch,random,os
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import ImageDraw, ImageFont

# 定义随机种子，实现代码复现
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载项目内训练好的分类模型
ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"checkpoints","resnet","best.pth") 
best_model = models.resnet18(weights=None)
best_model.fc = nn.Linear(best_model.fc.in_features,4)
'''resnet18默认fc层输出是1000维度，所以要改变输出维度的大小，和train定义fc一样'''
best_model.load_state_dict(torch.load(ckpt_path,map_location=device))
best_model = best_model.to(device).eval()
'''resnet18里面有大量的BatchNorm，但是训练集会使用单Batch进行归一化，只有添加eval才会使用训练的累积参数'''

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class_labels = ['overripe', 'ripe', 'rotten', 'unripe']

# 创建一个图像分类的模型，返回预测结果
def predict(image):
    if image is None:
        return None, None
    image = image.convert("RGB")
    inputs = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = best_model(inputs)
        prob = torch.softmax(outputs,dim=1)   # prob是置信度
        confidence,pred = torch.max(prob,1)   # confidence是预测概率，pred是预测类别
        labels = class_labels[pred.item()]
        confidence_score = confidence.item() * 100

    # 在图片上添加分类结果
    image_with_text = image.copy()
    draw = ImageDraw.Draw(image_with_text)

    # 设置字体
    font = ImageFont.truetype("arial.ttf", 25)  # Windows 可能有 Arial.ttf

    # 计算文本位置
    text = f"{labels} ({confidence_score:.2f}%)"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    image_width, _ = image_with_text.size
    text_x = (image_width - text_width) // 2
    text_y = 10 

    # 绘制文本
    draw.text((text_x, text_y), text, fill="red", font=font)

    return image, image_with_text

# 搭建Gradio界面
img_height = 400
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", height=img_height),
    outputs=[gr.Image(type="pil", label="原始图片", height=img_height), gr.Image(type="pil", label="鉴别结果", height=img_height)],
    title="香蕉图像鉴别系统",
    description="上传图片，右侧会显示原始图像和分类后的图像与预测类别",
)

# 如果inbrowser=False：不自动打开系统外部浏览器，改用 IDE 内置预览打开
iface.launch(inbrowser=True)