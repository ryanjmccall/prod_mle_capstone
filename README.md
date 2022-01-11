# prod_mle_project
Production Capstone Project for Springboard Machine Learning Engineer Career Track

## Installation

In order to support librosa, this package requires Python 3.7.*
The requirements.txt was generated from a conda environment.
- `conda create --name prod python=3.7 --file requirements.txt`
- `conda activate prod`

Install the `sentiment_classifier` Python package within this repo by running from repo root:

- `pip install -e .`

- `mkdir logs`

### Creating the requirements.txt
For reference the requirements.txt file was created with the following commands:
- `conda create --name prod python=3.7`
- Run `conda install -c conda-forge -y [key-libraries]`
- `conda list --export > requirements.txt`

## Testing

From repo root run:
`pytest tests`

## Running

Run the ML training pipeline from CL to prepare the data and train a model:
`python src/sentiment_classifier/pipeline.py`

Instead of training a model, run a Bayesian hyperparameter search:
`python src/sentiment_classifier/pipeline.py --search`

in either case, the results are stored to `data/results/*`

## Flask API 
To run flask locally
`export FLASK_APP='src/sentiment_classifier/prediction/wsgi.py' && export FLASK_ENV=development && flask run`

## Docker
From the repo root run:
`docker image build -t prediction_app .`

`docker run -d -p 5000:5000 prediction_app`

Then navigate in a browser to:
`localhost:5000`

## AWS 
Configure CLI: 
https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html


Use AWS lightsail: 
https://aws.amazon.com/getting-started/hands-on/serve-a-flask-app/

```
docker build -t flask-container .

aws lightsail create-container-service --service-name flask-service --power small --scale 1

aws lightsail push-container-image --service-name flask-service --label flask-container --image flask-container

# note container number in output

aws lightsail create-container-service-deployment \
--service-name flask-service \
--containers file://deploy/containers.json \
--public-endpoint file://deploy/public-endpoint.json

# check status
aws lightsail get-container-services --service-name flask-service

# Cleanup and delete Lightsail resources
aws lightsail delete-container-service --service-name flask-service

```


