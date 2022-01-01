# prod_mle_project
Production Capstone Project for Springboard Machine Learning Engineer Career Track

## Installation

In order to support librosa, this package requires Python 3.7.*
The requirements.txt was generated from a conda environment.
- `conda create --name py37 python=3.7 --file requirements.txt`
- `conda activate py37`
- (FYI `conda list --export > requirements.txt`)
- TODO: recreate the environment from scratch 
- to ensure minimality producing new requirements.txt

Install this Python package with 
`cd X/prod_mle_capstone && pip install -e .`

`mkdir logs`

`export PYTHONPATH=$PYTHONPATH:/Users/home/PycharmProjects/prod_mle_capstone`

## Testing

From repo root run:
`pytest tests`

## Running

Run the ML training pipeline from CL:
`python src/sentiment_classifier/pipeline.py`

Optionally, run a Bayes optimized hyperparameter search instead of model 
training/testing:
`python src/sentiment_classifier/pipeline.py --search`

## Docker

TBD

## Flask API with Gunicorn

TBD
