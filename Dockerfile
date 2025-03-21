FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

CMD ["python", "./main.py"]