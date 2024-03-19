import torch
import torch.nn as nn
from torchvision.models import resnet18
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torchvision.transforms as T
import json

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# Путь к файлу конфигурации и весам модели
weights_path_yolo = 'utils/best_yolo_weights.pt'


class ruBERTToxicClassifier(nn.Module):
    def __init__(self, model_class, pretrained_weights):
        super().__init__()
        self.bert = model_class.from_pretrained(pretrained_weights)
        # Замораживаем все параметры
        for param in self.bert.parameters():
            param.requires_grad = False

        # Размораживаем слой intermediate
        for param in self.bert.bert.encoder.layer[0].intermediate.parameters():
            param.requires_grad = True

        # Размораживаем слой output
        for param in self.bert.bert.encoder.layer[0].output.parameters():
            param.requires_grad = True

        # Размораживаем слой BertPooler
        for param in self.bert.bert.pooler.parameters():
            param.requires_grad = True

        # Размораживаем слой classifier
        for param in self.bert.classifier.parameters():
            param.requires_grad = True
                
        self.linear = nn.Sequential(
            nn.Linear(5, 256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )        

    def forward(self, x, attention_mask):
        bert_out = self.bert(x, attention_mask=attention_mask)[0]
        out = self.linear(bert_out)
        return out


def load_classes():
    '''
    Returns IMAGENET classes
    '''
    with open('utils/imagenet-simple-labels.json') as f:
        labels = json.load(f)
    return labels

def class_id_to_label(i):
    '''
    Input int: class index
    Returns class name
    '''

    labels = load_classes()
    return labels[i]

def load_model_yolo():
    # '''
    # Returns resnet model with IMAGENET weights
    # '''
    '''
    Returns YOLO8 model with trained weights
    '''
    # model = resnet18()
    # model.load_state_dict(torch.load('utils/resnet18-weights.pth', map_location='cpu'))
    # model.eval()

    '''
    Возвращает модель YOLO с загруженными весами.
    '''

    # Создание экземпляра модели без загрузки весов
    model = YOLO(weights_path_yolo)

    # Загрузка весов модели
    # model.load(weights_path)

    return model


def load_model_toxicity(model_class, pretrained_weights, weights_path):
    # Создаем экземпляр классификатора
    model = ruBERTToxicClassifier(model_class, pretrained_weights)

    # Загружаем веса
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))


    return model

def transform_image(img):
    '''
    Input: PIL img
    Returns: transformed image
    '''
    trnsfrms = T.Compose(
        [
            T.Resize((224, 224)), 
            T.CenterCrop(100),
            T.ToTensor(),
            T.Normalize(mean, std)
        ]
    )
    return trnsfrms(img)


