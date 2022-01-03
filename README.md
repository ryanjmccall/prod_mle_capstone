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
- Run `conda install -c conda-forge -y dask dask-ml imbalanced-learn joblib librosa lightgbm moviepy numpy pandas prefect pytest scikit-learn scikit-optimize`
- `conda list --export > requirements.txt`

## Testing

From repo root run:
`pytest tests`

## Running

Run the ML training pipeline from CL:
`python src/sentiment_classifier/pipeline.py`

# TODO argument to specify the path to the CONFIG

Optionally, run a Bayes optimized hyperparameter search instead of model 
training/testing:
`python src/sentiment_classifier/pipeline.py --search`

## Docker

TBD

## Flask API with Gunicorn

TBD
