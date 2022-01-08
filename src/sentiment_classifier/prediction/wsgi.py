import logging
from logging.config import dictConfig
import os
import time

from flask import Flask, request, render_template, jsonify, flash
from joblib import load
import librosa
from werkzeug.utils import secure_filename, redirect

from sentiment_classifier.prediction.log_config import PREDICTION_LOG_CONFIG
from sentiment_classifier.task.extract_helper import extract_features


# configure logging
logging.basicConfig(level=logging.INFO)
# dictConfig(PREDICTION_LOG_CONFIG)


_DIR_NAME = os.path.dirname(os.path.realpath(__file__))
_PIPELINE = load(os.path.join(_DIR_NAME, 'artifacts/prediction_pipeline.joblib'))
_ALLOWED_EXTENSIONS = {'wav', 'm4a'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(_DIR_NAME, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['SESSION_TYPE'] = 'filesystem'
# app.secret_key = 'super secret key'


@app.route('/', methods=['GET', 'POST'])
def post_redirect_get():
    app.logger.info('hit /')
    return redirect('upload', code=303)


@app.route('/upload')
def upload():
    app.logger.info('hit upload')
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    app.logger.info('hit uploader')
    if request.method == 'POST':
        if 'file' not in request.files:
            app.logger.info('No file part in request')
            flash('No file part in request')
            return redirect(request.url)

        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            app.logger.info('No selected file')
            flash('No selected file')
            return redirect(request.url)

        filename = secure_filename(uploaded_file.filename)
        app.logger.info('Checking if filename: %s is allowed', filename)
        if is_allowed_file(filename):
            start = time.time()
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(save_path)

            # hit errors, but seems I can get the file directly from uploaded_file.stream ?
            # https://werkzeug.palletsprojects.com/en/1.0.x/datastructures/#werkzeug.datastructures.FileStorage
            # audio, sr = librosa.load(uploaded_file.stream.read())

            # prediction = 0
            prediction = get_prediction(save_path)
            elapsed = round(time.time() - start, 6)
            app.logger.info('Computed prediction %s', prediction)
            return jsonify({'negative_sentiment_prediction': prediction,
                            'processing_time': elapsed})
        else:
            app.logger.info('Rejecting disallowed file %s', filename)

    return render_template('upload.html')


def is_allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in _ALLOWED_EXTENSIONS


def get_prediction(audio_path: str) -> int:
    audio, sr = librosa.load(audio_path)
    os.remove(audio_path)  # could be run async?
    features = extract_features(audio, sr)  # could be pulled into the Pipeline
    prediction = _PIPELINE.predict(features.reshape(1, -1))
    return prediction.tolist()[0]


if __name__ == '__main__':
    app.run()
