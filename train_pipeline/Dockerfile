FROM python:3.10

WORKDIR /work_dir

COPY ./requirements.txt /work_dir/requirements.txt

COPY . /work_dir/train_pipeline

RUN pip install --no-cache-dir --upgrade -r /work_dir/requirements.txt

CMD ["python", "train_pipeline/main.py"]

