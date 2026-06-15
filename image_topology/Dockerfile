FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docker_pipeline.py .

ENV IMAGES_DIR=/data/images
ENV OUTPUT_DIR=/data/output

CMD ["python", "docker_pipeline.py"]