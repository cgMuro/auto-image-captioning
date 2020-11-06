FROM python:3.8-slim
COPY ./app.py /deploy/
COPY ./predict.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./model.h5 /deploy/
COPY ./VGG16.h5 /deploy/
COPY ./tokenizer.pkl /deploy/
WORKDIR /deploy/
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["sh", "-c", "streamlit run --server.enableCORS false --server.port $PORT /deploy/app.py"]