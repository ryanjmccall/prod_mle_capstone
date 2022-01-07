from logging.config import dictConfig
import os
import time

from flask import Flask, request, render_template, jsonify, flash
from joblib import load
import librosa
from werkzeug.utils import secure_filename, redirect

from sentiment_classifier.prediction.log_config import PREDICTION_LOG_CONFIG
from sentiment_classifier.task.extract import extract_features


dictConfig(PREDICTION_LOG_CONFIG)
PATH_TO_HERE = os.path.dirname(os.path.realpath(__file__))


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(PATH_TO_HERE, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'


_PIPELINE = load(os.path.join(PATH_TO_HERE, 'artifacts/prediction_pipeline.joblib'))


_ALLOWED_EXTENSIONS = {'wav', 'm4a'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in _ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload')
def upload_file():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
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
        if allowed_file(filename):
            start = time.time()
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(save_path)

            # hit errors, but seems I can get the file directly from uploaded_file.stream ?
            # https://werkzeug.palletsprojects.com/en/1.0.x/datastructures/#werkzeug.datastructures.FileStorage
            # audio, sr = librosa.load(uploaded_file.stream.read())

            audio, sr = librosa.load(save_path)
            os.remove(save_path)
            features = extract_features(audio, sr)  # could be pulled into the Pipeline
            prediction = _PIPELINE.predict(features.reshape(1, -1)).tolist()
            elapsed = round(time.time() - start, 6)
            return jsonify({'audio_len': len(audio),
                            'audio_sr': sr,
                            'negative_sentiment_prediction': prediction[0],
                            'processing_time': elapsed})
        else:
            app.logger.info('Rejecting disallowed file %s', filename)

    return render_template('upload.html')


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=False)
