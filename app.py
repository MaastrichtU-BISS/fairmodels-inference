"""
FAIRmodels.org Inference Engine - Main Flask Application

This application provides a web interface for performing inference with models
from FAIRmodels.org. It dynamically generates input forms based on model metadata
and executes models in Docker containers.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
from metadata_handler import MetadataHandler
from docker_executor import DockerExecutor
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.route('/')
def index():
    """Landing page with option to enter model ID."""
    return render_template('index.html')


@app.route('/model/<model_id>')
def model_form(model_id):
    """
    Display the inference form for a specific model.
    
    Args:
        model_id: The unique identifier for the model from FAIRmodels.org
    """
    try:
        metadata_handler = MetadataHandler()
        metadata = metadata_handler.fetch_metadata(model_id)
        
        if not metadata:
            return render_template('error.html', 
                                 error="Could not fetch model metadata"), 404
        
        # Extract variable information for form generation
        variables = metadata_handler.extract_variables(metadata)
        model_name = metadata_handler.get_model_name(metadata)
        docker_image = metadata_handler.get_docker_image(metadata)
        
        return render_template('model_form.html',
                             model_id=model_id,
                             model_name=model_name,
                             variables=variables,
                             docker_image=docker_image)
    
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {str(e)}")
        return render_template('error.html', 
                             error=f"Error loading model: {str(e)}"), 500


@app.route('/inference/<model_id>', methods=['POST'])
def perform_inference(model_id):
    """
    Perform inference using the submitted form data.
    
    Args:
        model_id: The unique identifier for the model
    """
    try:
        # Get form data
        input_data = request.form.to_dict()
        
        # Fetch metadata to get Docker image
        metadata_handler = MetadataHandler()
        metadata = metadata_handler.fetch_metadata(model_id)
        docker_image = metadata_handler.get_docker_image(metadata)
        
        if not docker_image:
            return jsonify({'error': 'Docker image not found in metadata'}), 400
        
        # Execute inference in Docker container
        docker_executor = DockerExecutor()
        result = docker_executor.run_inference(docker_image, input_data)
        
        return render_template('result.html',
                             model_id=model_id,
                             input_data=input_data,
                             result=result)
    
    except Exception as e:
        logger.error(f"Error performing inference for {model_id}: {str(e)}")
        return render_template('error.html', 
                             error=f"Inference error: {str(e)}"), 500


@app.route('/api/inference/<model_id>', methods=['POST'])
def api_inference(model_id):
    """
    API endpoint for performing inference (returns JSON).
    
    Args:
        model_id: The unique identifier for the model
    """
    try:
        # Get JSON data
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Fetch metadata to get Docker image
        metadata_handler = MetadataHandler()
        metadata = metadata_handler.fetch_metadata(model_id)
        docker_image = metadata_handler.get_docker_image(metadata)
        
        if not docker_image:
            return jsonify({'error': 'Docker image not found in metadata'}), 400
        
        # Execute inference in Docker container
        docker_executor = DockerExecutor()
        result = docker_executor.run_inference(docker_image, input_data)
        
        return jsonify({
            'model_id': model_id,
            'input': input_data,
            'result': result,
            'status': 'success'
        })
    
    except Exception as e:
        logger.error(f"API inference error for {model_id}: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
