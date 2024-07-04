from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import librosa
import numpy as np
import pandas as pd
import joblib
import logging
from pydub import AudioSegment
import requests

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the base path to the models directory
models_dir = os.path.join(os.path.dirname(__file__), 'models')

# Load models with error handling
models = {}
model_files = {
    'soundrep_model.joblib': 'soundrep_model',
    'wordrep_model.joblib': 'wordrep_model',
    'prolongation_model.joblib': 'prolongation_model'
}

for file, name in model_files.items():
    try:
        models[name] = joblib.load(os.path.join(models_dir, file))
        logger.info(f"Loaded model: {name}")
    except Exception as e:
        logger.error(f"Failed to load model {name}: {e}")

# Load the original DataFrame to get column names
df = pd.read_csv(os.path.join(models_dir, 'sep28k-mfcc.csv'))
column_names = df.columns[-13:]

# Function to extract MFCC features
def extract_mfcc(file_path, max_pad_len=130):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)

    # Padding or trimming to ensure consistent length
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]

    return np.mean(mfccs.T, axis=0)

@app.route('/')
def index():
    return "Welcome to the Fluency App Backend!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400

    audio_url = data['url']
    temp_wav_path = os.path.join('temp', 'temp_audio.wav')

    try:
        # Download the audio file from the URL
        response = requests.get(audio_url)
        with open(temp_wav_path, 'wb') as f:
            f.write(response.content)

        # Process the audio file
        mfcc_features = extract_mfcc(temp_wav_path)
        mfcc_features = mfcc_features.reshape(1, -1)
        mfcc_df = pd.DataFrame(mfcc_features, columns=column_names)

        # Predict stuttering types
        soundrep_prediction = models['soundrep_model'].predict(mfcc_df)[0]
        wordrep_prediction = models['wordrep_model'].predict(mfcc_df)[0]
        prolongation_prediction = models['prolongation_model'].predict(mfcc_df)[0]

        stuttering_types = []
        if soundrep_prediction == 1:
            stuttering_types.append('Sound Repetition')
        if wordrep_prediction == 1:
            stuttering_types.append('Word Repetition')
        if prolongation_prediction == 1:
            stuttering_types.append('Prolongation')

        result = {
            'stuttering': bool(stuttering_types),
            'types': stuttering_types
        }

        return jsonify(result)
    except KeyError as e:
        app.logger.error(f"KeyError: {e}")
        return jsonify({'error': 'Error processing the prediction'}), 500
    except Exception as e:
        app.logger.error(f"Error processing the prediction: {e}")
        return jsonify({'error': 'Error processing the prediction'}), 500

if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
