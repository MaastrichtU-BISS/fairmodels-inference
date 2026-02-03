# Inferencing engine

This repository contains a Python Flask application to perform the actual inferencing for models available on FAIRmodels.org.

The Python Flask application will build a webpage to provide the input, based on the metadata description. This means it will perform a terminology lookup for the human-readable name of the variable and provide this in the actual webpage.For categorical variables, it will provide the items as listed in the metadata. For continuous variables, it will provide a numerical input, bound with the minimum and maximum values available in the model metadata.
An example of the model metadata is provided in [https://v3.fairmodels.org/instance/e46b07fd-fbd5-4466-9e5b-79dfa36d347d](https://v3.fairmodels.org/instance/e46b07fd-fbd5-4466-9e5b-79dfa36d347d) which can be retrieved as JSON-LD when giving the appropriate accept headers in the HTTP GET request.

The metadata contains a link to a docker image, which contains the actual model. This should work similarly as the FAIVOR Validator code [see here](https://github.com/MaastrichtU-BISS/FAIVOR-ML-Validator). Which means this inferencing engine should download the docker image, start the image, perform the inferencing, and provide the result back to the end-user.

## Project Structure

```
inferencing/
├── app.py                  # Main Flask application
├── metadata_handler.py     # Handles model metadata retrieval
├── docker_executor.py      # Executes models in Docker containers
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker image for the application
├── docker-compose.yml     # Docker Compose configuration
├── .env.example           # Environment variables template
├── templates/             # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── model_form.html
│   ├── result.html
│   └── error.html
└── static/                # Static files
    └── css/
        └── style.css

```

## Features

- **Dynamic Form Generation**: Automatically generates input forms based on model metadata from FAIRmodels.org
- **Metadata Parsing**: Retrieves and parses JSON-LD metadata including variable types, constraints, and human-readable labels
- **Docker Integration**: Downloads and executes model Docker images for inference
- **Web Interface**: User-friendly web interface for model selection and data input
- **API Endpoint**: RESTful API for programmatic access to inference functionality
- **Categorical Variables**: Dropdown menus with options from metadata
- **Continuous Variables**: Number inputs with min/max validation from metadata

## Installation

### Prerequisites

- Python 3.11+
- Docker
- Docker Compose (optional)

### Local Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd inferencing
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy environment variables:
```bash
cp .env.example .env
```

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Docker Deployment

1. Build and run with Docker Compose:
```bash
docker-compose up -d
```

2. Or build and run manually:
```bash
docker build -t inferencing-engine .
docker run -p 5000:5000 -v /var/run/docker.sock:/var/run/docker.sock inferencing-engine
```

## Usage

### Web Interface

1. Navigate to `http://localhost:5000`
2. Enter a model ID from FAIRmodels.org (e.g., `e46b07fd-fbd5-4466-9e5b-79dfa36d347d`)
3. Fill in the dynamically generated form with input values
4. Click "Run Inference" to get predictions

### API Endpoint

Make a POST request to `/api/inference/<model_id>` with JSON input data:

```bash
curl -X POST http://localhost:5000/api/inference/e46b07fd-fbd5-4466-9e5b-79dfa36d347d \
  -H "Content-Type: application/json" \
  -d '{"variable1": "value1", "variable2": 123}'
```

## How It Works

1. **Metadata Retrieval**: The application fetches model metadata from FAIRmodels.org using the provided model ID
2. **Form Generation**: Based on the metadata, it dynamically creates an HTML form with appropriate input types and constraints
3. **Docker Execution**: When the user submits the form, the application:
   - Downloads the Docker image specified in the metadata
   - Creates a temporary input JSON file
   - Runs the container with the input data
   - Retrieves the inference results
4. **Result Display**: The predictions are displayed to the user

## Architecture

- **Flask**: Web framework for routing and template rendering
- **metadata_handler.py**: Handles all metadata operations including fetching from FAIRmodels.org and parsing variables
- **docker_executor.py**: Manages Docker operations including image pulling and container execution
- **Templates**: Jinja2 templates for dynamic HTML generation
- **Static files**: CSS for styling the web interface

## Security Considerations

- Docker containers run with limited resources (CPU and memory)
- Network access is disabled in containers for security
- Input validation based on metadata constraints
- Temporary files are cleaned up after execution

## Future Enhancements

- User authentication and session management
- Caching of model metadata and Docker images
- Batch inference support
- Inference history and logging
- Advanced error handling and retry logic
- Support for different model formats and containers

## License

[Specify your license here]

## Contributing

[Specify contribution guidelines here]
