FROM python:3.10

WORKDIR /work_dir

COPY ./requirements.txt /work_dir/requirements.txt

COPY ./train_pipeline /work_dir/train_pipeline

RUN pip install --no-cache-dir --upgrade -r /work_dir/requirements.txt

COPY ./app /work_dir/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]