# Menggunakan image dasar Python 3.10
FROM python:3.10-slim

# Mengatur direktori kerja di dalam container
WORKDIR /code

# Menyalin file requirements dan menginstalnya
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Menyalin seluruh kode dan modelmu ke dalam container
COPY . /code

# Memberikan izin untuk membuat direktori cache Hugging Face
RUN mkdir -p /code/cache && chmod 777 /code/cache
ENV TRANSFORMERS_CACHE=/code/cache

# Menjalankan aplikasi
CMD ["python", "app.py"]