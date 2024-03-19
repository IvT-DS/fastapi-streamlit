FROM python:3.9

# Установка зависимостей для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./app /code/app
COPY ./utils /code/utils

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
