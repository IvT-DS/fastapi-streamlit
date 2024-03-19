import requests
import streamlit as st
import json
import io
import base64
from PIL import Image



def main():

    st.title("Object detection")

    image = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if image:
        st.image(image, caption="Original image")

    if st.button("Detect ships!") and image is not None:
        

        # Преобразование UploadedFile в объект Image PIL
        img = Image.open(image)

        # Если изображение в режиме RGBA (с альфа-каналом), конвертируем его в RGB
        if img.mode == 'RGBA':
            img = img.convert('RGB')
             
        # Сохранение объекта Image PIL в байтовом потоке
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        
        # Отправка изображения на сервер для детектирования
        files = {"file": buffer}
        # res = requests.post("http://127.0.0.1:8000/detect", files=files)

        # Изменим немного адрес для работы docker-compose.yaml
        res = requests.post("http://backend.docker:8000/detect", files=files)


        # st.write(f"Ошибка при детектировании. Статус код: {res.status_code}")
        # st.write("Текст ошибки:", res.text)

        if res.status_code == 200:
            response_data = res.json()
            img_data = response_data['image']
            img_bytes = base64.b64decode(img_data)
            img_result = Image.open(io.BytesIO(img_bytes))
            st.image(img_result, caption="Detected Ships")
        else:
            st.error("Ошибка при детектировании.")


    txt = st.text_input('here')

    if st.button('send'):
        dat = {'text' : txt}
        res = requests.post("http://127.0.0.1:8000/clf_text", json=dat)



if __name__ == '__main__':
    main()
