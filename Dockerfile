FROM python
WORKDIR /app

ADD trash_detection.py /app
ADD best.pt /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6

RUN pip install ultralytics torch opencv-python deep_sort_realtime
CMD ["python", "./trash_detection.py"]
