# Menggunakan image base TensorFlow Serving
FROM tensorflow/serving:latest

# Mengatur environment variable untuk nama model
ENV MODEL_NAME=smart-ats-model

# Menyalin folder model yang sudah berstatus BLESSED ke dalam kontainer
COPY ./serving_model_dir/smart-ats-model /models/smart-ats-model