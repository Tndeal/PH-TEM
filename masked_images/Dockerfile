FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY masking_run.py .

ENV IMAGES_DIR=/data/test_images
ENV OUTPUT_DIR=/data/output_2

CMD ["python", "masking_run.py"]