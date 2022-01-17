# Audio Sentiment Classifier Capstone Project

# Summary
This repo contains my Capstone Project for the 
[Machine Learning Engineer Career Track](https://www.springboard.com/courses/ai-machine-learning-career-track/) at Springboard.
This project addresses the business needs of a hypothetical call center which must quickly determine whether its calls require escalation / intervention due to customer dissatisfaction. 
For this project I selected an audio dataset and then performed data cleaning, wrangling, and exploratory data analysis. 
Next I performed feature selection/development, assessed several ML algorithms, and performed various hyperparameter searches.
(Jupyter Notebooks available under `src/notebooks`)
With a performant model in hand, I developed an ETL pipeline to reliably reproduce the 
training results from the initial data. Next wrote an API to accept audio data and respond with a binary sentiment classification.
Finally, I containerized the API and deployed it to AWS. 

# Usage
Navigate to [Prod Prediction App](https://flask-service.kma9dfq1a9nuc.us-west-2.cs.amazonlightsail.com/) in a browser

Click the 'Browse' button to upload a file using the file picker and then click submit. 
The file must be of .wav format and less than 16 mb is size. The response will contain a binary value
specifying whether the audio's sentiment is negative (value 1 returned) or positive (value 0 returned).

# Dataset
The MELD dataset may be downloaded directly from: https://affective-meld.github.io/ See 'Download Raw Data'.
In particular, this project uses the Raw video files and their associated labels.
Here is a [backup](https://drive.google.com/drive/folders/1MIOJ-vCP218ds9yZaewbrA_M7SlMrRrm?usp=sharing) in case the original is down.

# Development Installation
Python 3.7.* is required to support the librosa audio package. Due to dependency compability issues the
production prediction API runs with its own (minimal) set of requirements distinct from those of the ETL pipeline.

## Prodution Prediction API
The Production Prediction API can be installed from the top-level requirements.txt with conda:

`conda create --name prediction_app python=3.7 --file requirements.txt`

`conda activate prediction_app`

Install the `sentiment_classifier` Python package:

`pip install -e .`

Run the Flask API locally:

`export FLASK_APP='src/sentiment_classifier/prediction/wsgi.py' && export FLASK_ENV=development && flask run`

## Training ETL Pipeline
The training ETL Pipeline uses a separate set of requirements:

`conda create --name etl_pipe python=3.7 --file requirements/etl_requirements.txt`

`conda activate etl_pipe`

Run the unit tests:
`pytest tests`

Download the raw video files downloaded to the `data/raw/dev|train|test` directories.
Run the ML training pipeline to prepare the data and train a model:
`python src/sentiment_classifier/run_dag.py`

To run a Bayesian hyperparameter search instead of training a model:
`python src/sentiment_classifier/run_dag.py --search`

in either case, the results of the run are stored to `data/results/`

## Creating requirements.txt
For reference the requirements.txt file was created by:
`pip list --format=freeze > *.txt`

This command is used due to a bug in `pip freeze` to obtain requirements with pinned version numbers which is 
installable using either `pip` or `conda`.

# Dockerizing the Prediction App
Must have Docker engine set up and running.

Create docker image from the Dockerfile:
`docker image build -t prediction_app .`

Run a docker container using the image
`docker run -d -p 5000:5000 prediction_app`

Then navigate in a browser to:
`localhost:5000`

# AWS Deployment
Configure AWS CLI: 
https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html

AWS lightsail instructions: 
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

# Sphinx documentation

Using html theme: `conda install -y sphinx_rtd_theme`

Configure sphinx:
`sphinx-quickstart docs`

Generate docs:
`sphinx-apidoc -f -o docs/source src/sentiment_classifier`

`cd docs && make html && cd ..`

Generated docs are viewable from:
`docs/build/html/index.html`

