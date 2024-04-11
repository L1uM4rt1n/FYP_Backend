FROM python:3.11

WORKDIR /app

COPY . /app

COPY linear_combi.pkl /app/linear_combi.pkl
COPY scaler.pkl /app/scaler.pkl

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]