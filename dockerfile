# Use a slim TensorFlow image
FROM tensorflow/tensorflow:2.14.0

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir -r reqirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
