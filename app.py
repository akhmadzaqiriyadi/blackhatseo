#!/usr/bin/env python3

import json
import logging
import os
from flask import Flask, request, jsonify
from blackhat_seo_detector import BlackHatSEODetector, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load configuration
config_path = 'config.json'
config = load_config(config_path)
if config.get('verbose', False):
    logger.setLevel(logging.DEBUG)

# Initialize detector
try:
    detector = BlackHatSEODetector(
        model_path='model.pkl',
        vectorizer_path='vectorizer.pkl',
        use_selenium=config.get('use_selenium', False),
        use_bert=config.get('use_bert', False)
    )
    logger.info("BlackHatSEODetector initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize detector: {e}")
    raise

# API Endpoints

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict black hat SEO for a list of URLs.
    Expects JSON: {"urls": ["url1", "url2", ...]}
    Returns JSON with predictions.
    """
    try:
        data = request.get_json()
        if not data or 'urls' not in data:
            return jsonify({
                'error': 'Invalid input. Expected JSON with "urls" key containing a list of URLs.'
            }), 400

        urls = data['urls']
        if not isinstance(urls, list):
            return jsonify({
                'error': 'Invalid input. "urls" must be a list.'
            }), 400

        if not urls:
            return jsonify({
                'error': 'No URLs provided.'
            }), 400

        logger.info(f"Predicting for {len(urls)} URLs")
        results = detector.predict(urls)
        return jsonify({
            'status': 'success',
            'results': results
        }), 200

    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/explain', methods=['POST'])
def explain():
    """
    Explain prediction for a single URL.
    Expects JSON: {"url": "https://example.com"}
    Returns JSON with explanation.
    """
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                'error': 'Invalid input. Expected JSON with "url" key.'
            }), 400

        url = data['url']
        if not isinstance(url, str):
            return jsonify({
                'error': 'Invalid input. "url" must be a string.'
            }), 400

        logger.info(f"Explaining prediction for {url}")
        explanation = detector.explain_prediction(url)
        return jsonify({
            'status': 'success',
            'explanation': explanation
        }), 200

    except Exception as e:
        logger.error(f"Error in /explain: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    Returns JSON indicating API status.
    """
    return jsonify({
        'status': 'healthy',
        'message': 'API is running'
    }), 200

# Run the app
if __name__ == '__main__':
    host = config.get('host', '0.0.0.0')
    port = config.get('port', 5001)
    debug = config.get('debug', False)
    logger.info(f"Starting Flask API on {host}:{port}")
    app.run(host=host, port=port, debug=debug)