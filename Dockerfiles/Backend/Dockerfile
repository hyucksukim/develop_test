# FROM python:3.9-buster

# COPY ./app /app
# COPY ./requirements.txt /app
# WORKDIR /app

# ENV	PYTHONFAULTHANDLER=1 \
# 	PYTHONUNBUFFERED=1 \
# 	PYTHONHASHSEED=random \
# 	PIP_NO_CACHE_DIR=off \
# 	PIP_DISABLE_PIP_VERSION_CHECK=on \
# 	PIP_DEFAULT_TIMEOUT=100

# #RUN apt-get install -y build-essential libpython
# RUN pip install -r requirements.txt
# #CMD ["uvicorn", "app.py:app", "--host", "localhost", "--port", "8080", "--reload"]
# CMD ["python3", "app.py"]


FROM python:3.9-buster

ENV	PYTHONFAULTHANDLER=1 \
	PYTHONUNBUFFERED=1 \
	PYTHONHASHSEED=random \
	PIP_NO_CACHE_DIR=off \
	PIP_DISABLE_PIP_VERSION_CHECK=on \
	PIP_DEFAULT_TIMEOUT=100

RUN mkdir -p /app
RUN mkdir -p "/workspace/data/boostcamp/"
COPY ./requirements.txt /app
WORKDIR /app
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY ./app/ /app/
RUN cp /app/boostcamp.inter /workspace/data/boostcamp/
# COPY ./app/MacridVAE-cpu_final.pth /app/saved/
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
