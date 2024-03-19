
# Добавление каталога в sys.path
import sys
sys.path.append('/home/ivant/DS_bootcamp/Phase3_lec_hw/Week1/Day1/HW/fastapi-streamlit/api')

# Импорт библиотек
import PIL
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from utils.model_func import class_id_to_label, load_model_yolo, load_model_toxicity, transform_image
import cv2
import numpy as np
import base64
from io import BytesIO
from fastapi.responses import JSONResponse
from PIL import Image
import io
# импортируем трансформеры
import transformers

# Указываем классы модели, токенайзера и стандартные веса
model_class = transformers.AutoModelForSequenceClassification
tokenizer_class = transformers.AutoTokenizer
pretrained_weights = 'cointegrated/rubert-tiny-toxicity'
weights_path_text = "utils/best_bert_tox_weights.pth"

model_yolo = None 
model_rubert = None
app = FastAPI()


# Create class of answer: only class name 
class ImageClass(BaseModel):
    prediction: str

class TextClass(BaseModel):
    text: str



# Load model at startup
@app.on_event("startup")
def startup_event():
    global model_yolo, model_rubert
    model_yolo = load_model_yolo()
    model_rubert = load_model_toxicity(model_class, pretrained_weights, weights_path_text)


@app.get('/')
def return_info():
    return 'Hello FastAPI'


# @app.post('/classify')
# def classify(file: UploadFile = File(...)):
#     image = PIL.Image.open(file.file)
#     adapted_image = transform_image(image)
#     pred_index = model(adapted_image.unsqueeze(0)).detach().cpu().numpy().argmax()
#     imagenet_class = class_id_to_label(pred_index)
#     response = ImageClass(
#         prediction=imagenet_class
#     )
#     return response

@app.post('/clf_text')
def clf_text(data: TextClass):
    print(data.text)
    return data


# Функция для рисования bounding boxes
def draw_boxes(image, detections):
    for detection in detections:
        if detection.boxes is not None and len(detection.boxes.data) > 0:
            # Итерация по каждой детекции
            for i in range(len(detection.boxes.data)):
                box = detection.boxes.xyxy[i].cpu().numpy()  # Координаты рамки
                conf = detection.boxes.conf[i].item()  # Уверенность
                cls = detection.boxes.cls[i].item()  # Класс
                
                # Преобразование координат рамки в целочисленные значения
                x1, y1, x2, y2 = map(int, box)
                
                # Рисуем рамку на изображении
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Добавляем текст (например, класс и уверенность)
                label = f'{detection.names[int(cls)]}: {conf:.2f}'
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 255), thickness=2)
    return image

@app.post('/detect')
async def detect_objects(file: UploadFile = File(...)):
    # Чтение изображения
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Вызов модели для получения детекций
    # Здесь предполагается, что detections - это список словарей с координатами bounding boxes
    detections = model_yolo.predict(image)

    # Рисование bounding boxes на изображении
    # test_detections = {'boxes': [(100, 100, 200, 200, 0.9, 1)]}
    image_with_boxes = draw_boxes(image, detections)

    # Конвертация обратно в формат, пригодный для отправки через HTTP
    _, buffer = cv2.imencode('.jpg', image_with_boxes)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse(content={"image": img_str})


 



##### run from api folder:
##### uvicorn app.main:app