import json
import plotly
import pandas as pd
import os
import gdown
import requests
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
from joblib import load
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    """
    Tokenizes and processes text data.

    Args:
    text (str): Text to be tokenized.

    Returns:
    list: List of cleaned tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens

# Load data
database_filepath = os.path.abspath(os.path.join(os.getcwd(), '../data/DisasterResponse.db'))
engine = create_engine(f'sqlite:///{database_filepath}')
df = pd.read_sql_table('disaster_messages', engine)

# Google Drive model file ID
file_id = '1eMAjZM3_oCC_cV-EVUswCnL3_jj31ryH'
model_filepath = os.path.abspath(os.path.join(os.getcwd(), '../models/classifier.pkl'))
download_url = f'https://drive.google.com/uc?id={file_id}'

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(model_filepath):
        print(f"Model file does not exist locally. Downloading model from Google Drive...")
        try:
            gdown.download(download_url, model_filepath, quiet=False)
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading the model using gdown: {e}")
            print("Attempting direct download method...")
            download_from_drive(file_id, model_filepath)
    else:
        print("Model file already exists locally.")

# Function to download file using direct method (if gdown fails)
def download_from_drive(file_id, destination):
    URL = f"https://drive.google.com/uc?id={file_id}"
    try:
        response = requests.get(URL, stream=True)
        if response.status_code == 200:
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"Model downloaded to {destination}")
        else:
            print(f"Failed to download file. HTTP Status Code: {response.status_code}")
    except Exception as e:
        print(f"Error during direct download: {e}")

# Download model
download_model()

# Verify downloaded file size (allowing a small tolerance)
expected_file_size = 1_100_000_000  # Approximate size of classifier.pkl in bytes
tolerance = 5 * 1024 * 1024  # Allow a tolerance of 5MB
actual_file_size = os.path.getsize(model_filepath)

if abs(actual_file_size - expected_file_size) > tolerance:
    print(f"Downloaded file is incomplete or corrupted. Expected size: {expected_file_size}, Got: {actual_file_size}")
    exit(1)

# Load model
try:
    print(f"Loading model from: {model_filepath}")
    model = load(model_filepath)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Home page route
@app.route('/')
@app.route('/index')
def index():
    """
    Render the home page with visualizations.
    """
    # Extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Create visuals
    graphs = [
        {
            'data': [Bar(x=genre_names, y=genre_counts)],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        }
    ]
    
    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Handle user input and display classification results.
    """
    # Get user input in query
    query = request.args.get('query', '') 

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render the go.html page with results
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

# Run the app
def main():
    """
    Run the Flask web app.
    """
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()
