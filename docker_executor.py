"""
Docker Executor Module

Handles downloading Docker images and running inference in containers.
Based on the FAIVOR ML Validator approach.
"""

import docker
import json
import logging
import requests
import time
import socket
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class DockerExecutor:
    """Handles Docker container execution for model inference."""
    
    def __init__(self):
        """Initialize Docker client."""
        try:
            self.client = docker.from_env()
            # Test connection
            self.client.ping()
            logger.info("Docker client initialized successfully")
            
            # Detect current container's network(s)
            self.networks = self._detect_current_networks()
            logger.info(f"Detected networks: {self.networks}")
        except docker.errors.DockerException as e:
            logger.error(f"Failed to initialize Docker client: {str(e)}")
            logger.error("Please ensure Docker is installed and running")
            logger.error("You may need to run: sudo systemctl start docker")
            raise Exception(
                "Docker is not available. Please ensure Docker is installed and running. "
                "You may need to start the Docker service or check permissions."
            )
        except Exception as e:
            logger.error(f"Unexpected error initializing Docker client: {str(e)}")
            raise
    
    def _detect_current_networks(self) -> List[str]:
        """
        Detect which Docker networks the current container is connected to.
        Returns a list of network names or IDs.
        """
        try:
            # Get current hostname (container ID when running in a container)
            hostname = socket.gethostname()
            logger.info(f"Current hostname: {hostname}")
            
            # Try to find this container
            try:
                container = self.client.containers.get(hostname)
                networks = list(container.attrs['NetworkSettings']['Networks'].keys())
                logger.info(f"Found current container {hostname} in networks: {networks}")
                return networks
            except docker.errors.NotFound:
                logger.info("Not running inside a container, or container not found")
                return []
            except Exception as e:
                logger.warning(f"Could not detect container networks: {str(e)}")
                return []
        except Exception as e:
            logger.warning(f"Error detecting networks: {str(e)}")
            return []
    
    def run_inference(self, docker_image: str, input_data: Dict[str, Any]) -> Any:
        """
        Run inference by executing the model in a Docker container.
        
        Args:
            docker_image: The Docker image URL/name
            input_data: Dictionary containing input variable values
            
        Returns:
            The inference result
        """
        logger.info(f"Starting inference with image: {docker_image}")
        logger.info(f"Input data: {input_data}")
        
        container = None
        try:
            # Pull the Docker image
            self._pull_image(docker_image)
            
            # Run the container with HTTP server
            container = self._start_container(docker_image)
            
            # Wait for the server to be ready
            base_url = self._wait_for_server(container)
            
            # Make HTTP request to the model
            result = self._make_inference_request(base_url, input_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise
        finally:
            # Clean up: stop and remove container
            if container:
                try:
                    logger.info(f"Stopping container {container.id[:12]}")
                    container.stop(timeout=5)
                    container.remove()
                    logger.info("Container stopped and removed")
                except Exception as e:
                    logger.warning(f"Error cleaning up container: {str(e)}")
    
    def _pull_image(self, image_name: str) -> None:
        """
        Pull the Docker image if not already available locally.
        
        Args:
            image_name: The Docker image name/URL
        """
        try:
            logger.info(f"Checking for image: {image_name}")
            # Check if image exists locally
            try:
                self.client.images.get(image_name)
                logger.info(f"Image {image_name} already available locally")
                return
            except docker.errors.ImageNotFound:
                pass
            
            # Pull the image
            logger.info(f"Pulling image: {image_name}")
            self.client.images.pull(image_name)
            logger.info(f"Successfully pulled image: {image_name}")
        
        except Exception as e:
            logger.error(f"Failed to pull image {image_name}: {str(e)}")
            raise
    
    def _start_container(self, image_name: str):
        """
        Start the Docker container with the model server.
        
        Args:
            image_name: The Docker image name
            
        Returns:
            The running container object
        """
        try:
            logger.info(f"Starting container with image: {image_name}")
            
            # Determine network configuration
            network_config = None
            if self.networks:
                # Use the first detected network
                network_config = self.networks[0]
                logger.info(f"Connecting model container to network: {network_config}")
            
            # Start container in detached mode with port mapping
            container = self.client.containers.run(
                image_name,
                detach=True,
                ports={'8000/tcp': None},  # Map to random host port
                network=network_config,  # Connect to same network
                remove=False,  # We'll remove it manually
                mem_limit='1g',
                cpu_period=100000,
                cpu_quota=50000
            )
            
            logger.info(f"Container started with ID: {container.id[:12]}")
            
            # Connect to additional networks if present
            if len(self.networks) > 1:
                for network in self.networks[1:]:
                    try:
                        network_obj = self.client.networks.get(network)
                        network_obj.connect(container)
                        logger.info(f"Connected container to additional network: {network}")
                    except Exception as e:
                        logger.warning(f"Could not connect to network {network}: {str(e)}")
            
            return container
            
        except Exception as e:
            logger.error(f"Failed to start container: {str(e)}")
            raise
    
    def _wait_for_server(self, container, timeout: int = 60) -> str:
        """
        Wait for the server inside the container to be ready.
        
        Args:
            container: The Docker container object
            timeout: Maximum time to wait in seconds
            
        Returns:
            The base URL where the server is accessible (either container IP or localhost:port)
        """
        # Reload container to get network information
        container.reload()
        
        # Determine the base URL based on whether we're running in a container
        base_url = None
        if self.networks:
            # We're running in a container - use container's internal network address
            network_name = self.networks[0]
            network_settings = container.attrs['NetworkSettings']['Networks'].get(network_name)
            if network_settings and network_settings.get('IPAddress'):
                container_ip = network_settings['IPAddress']
                base_url = f"http://{container_ip}:8000"
                logger.info(f"Using container internal address: {base_url}")
            else:
                # Fallback: use container name or ID
                container_name = container.name
                base_url = f"http://{container_name}:8000"
                logger.info(f"Using container name: {base_url}")
        else:
            # Running locally - use localhost with mapped port
            port_mapping = container.ports.get('8000/tcp')
            if not port_mapping:
                raise Exception("Container did not expose port 8000")
            host_port = int(port_mapping[0]['HostPort'])
            base_url = f"http://localhost:{host_port}"
            logger.info(f"Using localhost with mapped port: {base_url}")
        
        # Wait for server to be ready
        start_time = time.time()
        last_error = None
        
        while time.time() - start_time < timeout:
            try:
                # Try multiple endpoints to check if server is up
                for endpoint in ['/health', '/docs', '/']:
                    try:
                        response = requests.get(
                            f"{base_url}{endpoint}", 
                            timeout=2
                        )
                        # Any response (even 404) means server is running
                        if response.status_code in [200, 404, 405]:
                            logger.info(f"Server is ready (checked {endpoint})")
                            return base_url
                    except requests.exceptions.RequestException as e:
                        last_error = str(e)
                        continue
            except Exception as e:
                last_error = str(e)
            
            # Check if container is still running
            container.reload()
            if container.status != 'running':
                logs = container.logs().decode('utf-8')
                raise Exception(f"Container stopped unexpectedly. Logs:\n{logs}")
            
            time.sleep(1)
        
        # Get container logs for debugging
        logs = container.logs().decode('utf-8')
        raise Exception(
            f"Server did not become ready within {timeout} seconds. "
            f"Last error: {last_error}\nContainer logs:\n{logs}"
        )
    
    def _make_inference_request(self, base_url: str, input_data: Dict[str, Any]) -> Any:
        """
        Make HTTP request to the model server for inference.
        
        Args:
            base_url: The base URL where the server is running
            input_data: Dictionary containing input values
            
        Returns:
            The parsed inference result
        """
        try:
            
            # Step 1: POST to /predict to start the prediction
            logger.info(f"Starting prediction request to {base_url}/predict")
            logger.info(f"Request payload: {input_data}")
            
            response = requests.post(
                f"{base_url}/predict",
                json=input_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            # /predict doesn't return data, it just starts the process
            if response.status_code not in [200, 204]:
                logger.error(f"Predict request failed: {response.status_code} - {response.text}")
                raise Exception(f"Prediction request failed: {response.text}")
            
            logger.info("Prediction started, checking status...")
            
            # Step 2: Poll /status until prediction is complete
            max_wait = 60  # seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                status_response = requests.get(f"{base_url}/status", timeout=5)
                status_response.raise_for_status()
                status_data = status_response.json()
                
                status_id = status_data.get('status')
                message = status_data.get('message', '')
                
                logger.info(f"Status: {status_id} - {message}")
                
                if status_id == 3:  # Prediction completed
                    logger.info("Prediction completed successfully")
                    break
                elif status_id == 4:  # Prediction failed
                    error_msg = status_data.get('message', 'Unknown error')
                    raise Exception(f"Model prediction failed: {error_msg}")
                
                time.sleep(1)
            else:
                raise Exception(f"Prediction did not complete within {max_wait} seconds")
            
            # Step 3: GET /result to retrieve the prediction
            logger.info(f"Fetching result from {base_url}/result")
            result_response = requests.get(f"{base_url}/result", timeout=5)
            result_response.raise_for_status()
            result = result_response.json()
            
            logger.info(f"Inference result: {result}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error making inference request: {str(e)}")
            raise Exception(f"Inference request failed: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error making inference request: {str(e)}")
            raise
    
    def cleanup(self):
        """Clean up Docker resources."""
        try:
            # Optionally prune unused containers and images
            # self.client.containers.prune()
            pass
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
