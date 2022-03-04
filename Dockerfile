FROM python:3.9-slim

COPY ./ ./
# WORKDIR /app

#Pip installs
RUN sudo apt-get install python3-opencv

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r "requirements.txt"

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "BE.py"]
# CMD ["streamlit run BE.py"]