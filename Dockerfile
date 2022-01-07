# set base image (host OS)
FROM python:3.7

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt
RUN pip install -e git+https://github.com/ryanjmccall/prod_mle_capstone.git#egg=sentiment_classifier-ryanmccall

# https://stackoverflow.com/questions/61235346/librosa-raised-oserrorsndfile-library-not-found-in-docker
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends build-essential gcc libsndfile1

# copy the content of the local src directory to the working directory
COPY src/ .

# command to run on container start
CMD ["python", "sentiment_classifier/prediction/wsgi.py" ]
