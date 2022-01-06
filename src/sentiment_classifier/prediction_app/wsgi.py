import os

import librosa
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename

from sentiment_classifier.task.extract import extract_features

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'wav', 'txt'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000

app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'


def _allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload')
def upload_file():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)

        # TODO call _allowed_file m4a, wav etc.

        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(save_path)

        # TODO can I simply take the file from the request object?
        audio, sr = librosa.load(save_path)

        features = extract_features(audio, sr)

        # TODO I have to pickle the fitted standardizer, PCA, and model then load them here
        # and run them all?
        # ('standardize', QuantileTransformer(**conf['standardize'])),
        # ('decomposition', PCA(**conf['decomposition'])),
        # ('model', lgb.LGBMClassifier(**conf['model'])),

        return jsonify({'negative': True, 'audio_len': len(audio), 'sr': sr,
                        'features': features.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
