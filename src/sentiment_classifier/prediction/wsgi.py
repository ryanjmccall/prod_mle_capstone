import logging
import os
import time

from flask import Flask, request, render_template, jsonify, flash
from joblib import load
import librosa
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename, redirect

from sentiment_classifier.task.extract_helper import extract_features


logging.basicConfig(level=logging.INFO)


_DIR_NAME = os.path.dirname(os.path.realpath(__file__))


# Load the ML pipeline from disk
_PIPELINE = load(os.path.join(_DIR_NAME, 'artifacts/prediction_pipeline.joblib'))


app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['UPLOAD_EXTENSIONS'] = ['wav', 'm4a']
app.config['UPLOAD_PATH'] = os.path.join(_DIR_NAME, 'uploads')


@app.route('/')
def index():
    app.logger.info('hit index /')
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    app.logger.info('hit upload_file /')
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
    if not allowed_file(filename):
        app.logger.info('Rejecting disallowed file %s', filename)
        abort(400)

    start = time.time()
    save_path = os.path.join(app.config['UPLOAD_PATH'], filename)
    uploaded_file.save(save_path)
    prediction = get_prediction(save_path)
    elapsed = round(time.time() - start, 6)
    app.logger.info('Computed prediction %s', prediction)
    return jsonify({'negative_sentiment_prediction': prediction,
                    'processing_time_sec': elapsed})

# hit errors, but seems I can get the file directly from uploaded_file.stream ?
# https://werkzeug.palletsprojects.com/en/1.0.x/datastructures/#werkzeug.datastructures.FileStorage
# audio, sr = librosa.load(uploaded_file.stream.read())


def allowed_file(filename: str) -> bool:
    if filename == '':
        return False

    if '.' not in filename:
        return False

    ext = filename.rsplit('.', 1)[1].lower()
    return ext in app.config['UPLOAD_EXTENSIONS']


def get_prediction(audio_path: str) -> int:
    audio, sr = librosa.load(audio_path)
    os.remove(audio_path)  # could be run async?
    features = extract_features(audio, sr)  # could be pulled into the Pipeline
    prediction = _PIPELINE.predict(features.reshape(1, -1))
    return prediction.tolist()[0]


if __name__ == '__main__':
    app.run()
