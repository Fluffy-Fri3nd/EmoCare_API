# Gunakan base image Python 3.9
FROM python:3.9

# Set environment variable untuk cache Transformers
ENV TRANSFORMERS_CACHE=/app/.cache

# Tentukan direktori kerja di dalam container
WORKDIR /code

# Salin file requirements.txt dan install dependencies
COPY ./requirements.txt /code/requirements.txt

# Install dependencies tanpa cache
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Salin seluruh proyek ke dalam container
COPY . .

# Pastikan folder cache sudah ada dan memiliki izin yang sesuai
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

# Perintah untuk menjalankan aplikasi FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
