# base image
FROM python:3.7

EXPOSE 5000/tcp

# set container working directory
WORKDIR /workdir

# https://stackoverflow.com/questions/61235346/librosa-raised-oserrorsndfile-library-not-found-in-docker
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends build-essential gcc libsndfile1

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install -e git+https://github.com/ryanjmccall/prod_mle_capstone.git#egg=sentiment_classifier-ryanmccall

# copy local src directory to workdir
COPY src/ .

ENV FLASK_APP=sentiment_classifier/prediction/wsgi.py
ENV FLASK_ENV=development
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# command to run on container start
CMD ["flask", "run"]
