FROM python:3.9-buster

COPY ./app /app
COPY ./requirements.txt /app
WORKDIR /app

ENV	PYTHONFAULTHANDLER=1 \
	PYTHONUNBUFFERED=1 \
	PYTHONHASHSEED=random \
	PIP_NO_CACHE_DIR=off \
	PIP_DISABLE_PIP_VERSION_CHECK=on \
	PIP_DEFAULT_TIMEOUT=100

#RUN apt-get install -y build-essential libpython
RUN pip install -r requirements.txt

EXPOSE 8080

#CMD ["uvicorn", "app.py:app", "--host", "localhost", "--port", "8080", "--reload"]
CMD ["python3", "app.py"]
