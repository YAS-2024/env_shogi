import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CUDAを無効にする
# PyTorchのCUDA無効化を明示的に設定
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle

model_path=os.getcwd() + '/models/resnet_model/resnet_shogi_model_qtver.pth'
transform_path=os.getcwd() + '/models/resnet_model/transform_128ver.pkl'

# クラスIDリスト
class_ids = [
    '011', '012', '021', '022', '031', '032', '041', '042',
    '051', '052', '061', '062', '071', '072', '081', '082',
    '111', '121', '131', '141', '171', '172', '181', '182'
]
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
num_classes = len(class_ids)

class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2Classifier, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

def predict_pices_class_from_image(image_path):
    # シリアライズされた前処理をロード
    with open(transform_path, 'rb') as f:
        transform = pickle.load(f)
    # 保存済みモデルのロード
    model = MobileNetV2Classifier(num_classes=num_classes)
    ###cuda無効に対する修正###
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    ###ここまで###

    model = model.to(device)
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return int(class_ids[preds.item()][1:])

def predict_pices_class_from_image_batch(image_paths):
    # シリアライズされた前処理をロード
    with open(transform_path, 'rb') as f:
        transform = pickle.load(f)
    # 保存済みモデルのロード
    model = MobileNetV2Classifier(num_classes=num_classes)
    
    ###cuda無効に対する修正###
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    ###ここまで###
    model = model.to(device)
    
    model.eval()
    
    #images = [transform(Image.open(img_path).convert("RGB")) for img_path in image_paths]
    images = [transform(img_instance.convert("RGB")) for img_instance in image_paths] #イメージインスタンスを直接渡す前提に変更    
    images = torch.stack(images).to(device)
    with torch.set_grad_enabled(False):
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    # 3桁の文字列の左から2桁を数値として返却        
    return [int(class_ids[p][:2]) for p in preds.cpu().numpy()]

if __name__ == '__main__':
    # 推論例
    filename='gote_20240616122615_00002.png'
    image_path = os.getcwd() + '/input/piece_classification task/' + filename
    #predicted_class = predict_pices_class_from_image(image_path)
    #print(f'Predicted class: {predicted_class}')

    # バッチ推論例
    # 高速化が必要な場合検討
    #image_paths = []
    image_paths = [os.getcwd() + '/input/piece_classification task/' + filename,os.getcwd() + '/input/piece_classification task/' + filename]
    predicted_classes = predict_pices_class_from_image_batch(image_paths)
    print(f'Predicted classes: {predicted_classes}')




