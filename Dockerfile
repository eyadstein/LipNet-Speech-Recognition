FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY app.py predict.py ./
COPY models/checkpoint.weights.h5 ./models/
COPY data/s1/bbal6n.mpg ./sample.mpg

EXPOSE 7860

CMD ["python", "app.py"]
