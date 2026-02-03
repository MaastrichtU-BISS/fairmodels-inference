"""
Metadata Handler Module

Handles fetching and parsing model metadata from FAIRmodels.org.
"""

import requests
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class MetadataHandler:
    """Handles fetching and parsing model metadata from FAIRmodels.org."""
    
    BASE_URL = "https://v3.fairmodels.org/instance/"
    LIST_URL = "https://v3.fairmodels.org/"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/ld+json'
        })
    
    def fetch_models_list(self) -> List[Dict[str, Any]]:
        """
        Fetch the list of available models from FAIRmodels.org.
        
        Returns:
            List of model dictionaries with id and metadata
        """
        try:
            logger.info(f"Fetching models list from {self.LIST_URL}")
            response = self.session.get(self.LIST_URL, timeout=10)
            response.raise_for_status()
            
            models_data = response.json()
            logger.info(f"Successfully fetched models list")
            
            # Parse the response - API returns dict where keys are UUIDs
            models = []
            if isinstance(models_data, dict):
                for model_id, model_info in models_data.items():
                    if isinstance(model_info, dict):
                        # Extract title from top-level or properties
                        title = model_info.get('title')
                        if not title and 'properties' in model_info:
                            props = model_info['properties']
                            title = props.get('General Model Information.Title')
                        
                        if not title:
                            title = model_id  # Fallback to ID
                        
                        models.append({
                            'id': model_id,
                            'title': title,
                            'raw': model_info
                        })
            
            logger.info(f"Parsed {len(models)} models from list")
            return models
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching models list: {str(e)}")
            return []
    
    def _extract_title_from_list_item(self, model: Dict[str, Any]) -> str:
        """Extract title from a model list item."""
        # Try various possible title fields
        for field in ['title', 'name', 'label']:
            if field in model:
                title = model[field]
                if isinstance(title, dict):
                    return title.get('@value', str(title))
                return str(title)
        
        # Try nested General Model Information
        if 'General Model Information' in model:
            info = model['General Model Information']
            if isinstance(info, dict):
                title = info.get('Title', {})
                if isinstance(title, dict):
                    return title.get('@value', '')
        
        return "Unknown Model"
    
    def fetch_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch model metadata from FAIRmodels.org.
        
        Args:
            model_id: The unique identifier for the model
            
        Returns:
            Dictionary containing the model metadata, or None if fetch fails
        """
        url = f"{self.BASE_URL}{model_id}"
        
        try:
            logger.info(f"Fetching metadata from {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            metadata = response.json()
            logger.info(f"Successfully fetched metadata for model {model_id}")
            return metadata
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching metadata for {model_id}: {str(e)}")
            return None
    
    def get_model_name(self, metadata: Dict[str, Any]) -> str:
        """
        Extract the model name from metadata.
        
        Args:
            metadata: The model metadata dictionary
            
        Returns:
            The model name or a default value
        """
        # Check FAIRmodels format first
        if 'General Model Information' in metadata:
            general_info = metadata['General Model Information']
            if isinstance(general_info, dict):
                title = general_info.get('Title', {})
                if isinstance(title, dict):
                    title_value = title.get('@value', '')
                    if title_value:
                        return title_value
                elif title:
                    return str(title)
        
        # Try different possible fields for the name
        for field in ['name', 'title', 'label', '@id']:
            if field in metadata:
                return str(metadata[field])
        
        return "Unknown Model"
    
    def get_docker_image(self, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Extract the Docker image URL from metadata.
        
        Args:
            metadata: The model metadata dictionary
            
        Returns:
            The Docker image URL or None
        """
        # Check FAIRmodels format first
        if 'General Model Information' in metadata:
            general_info = metadata['General Model Information']
            if isinstance(general_info, dict):
                fairmodels_image = general_info.get('FAIRmodels image name', {})
                if isinstance(fairmodels_image, dict):
                    image_name = fairmodels_image.get('@value', '')
                    if image_name:
                        return image_name
                elif fairmodels_image:
                    return fairmodels_image
        
        # Look for Docker image in various possible locations
        if 'implementation' in metadata:
            impl = metadata['implementation']
            if isinstance(impl, dict) and 'dockerImage' in impl:
                return impl['dockerImage']
            elif isinstance(impl, str):
                return impl
        
        if 'dockerImage' in metadata:
            return metadata['dockerImage']
        
        if 'container' in metadata:
            container = metadata['container']
            if isinstance(container, dict) and 'image' in container:
                return container['image']
            elif isinstance(container, str):
                return container
        
        logger.warning("Docker image not found in metadata")
        return None
    
    def extract_variables(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract input variables from metadata.
        
        Args:
            metadata: The model metadata dictionary
            
        Returns:
            List of variable definitions with name, type, constraints, etc.
        """
        variables = []
        
        # Look for variables in different possible locations
        var_sources = [
            metadata.get('Input data1', []),  # FAIRmodels.org format
            metadata.get('Input data', []),
            metadata.get('variables', []),
            metadata.get('inputs', []),
            metadata.get('features', []),
            metadata.get('parameters', [])
        ]
        
        for source in var_sources:
            if source:
                variables.extend(self._parse_variables(source))
        
        # If still no variables found, look in schema
        if not variables and 'schema' in metadata:
            schema = metadata['schema']
            if isinstance(schema, dict) and 'properties' in schema:
                variables = self._parse_schema_properties(schema['properties'])
        
        return variables
    
    def _parse_variables(self, var_list: List[Any]) -> List[Dict[str, Any]]:
        """Parse a list of variable definitions."""
        parsed = []
        
        for var in var_list:
            if not isinstance(var, dict):
                continue
            
            # Handle FAIRmodels.org format with nested structure
            input_label = var.get('Input label', {})
            if isinstance(input_label, dict):
                input_label = input_label.get('@value', '')
            
            description_obj = var.get('Description', {})
            if isinstance(description_obj, dict):
                description = description_obj.get('@value', '')
            else:
                description = var.get('description', '')
            
            # Get the feature label for human-readable name
            input_feature = var.get('Input feature', {})
            if isinstance(input_feature, dict):
                rdfs_label = input_feature.get('rdfs:label', {})
                if isinstance(rdfs_label, dict):
                    feature_label = rdfs_label.get('@value', '')
                else:
                    feature_label = rdfs_label
            else:
                feature_label = ''
            
            variable_info = {
                'name': input_label or var.get('name', var.get('id', 'unknown')),
                'label': feature_label or self._get_human_readable_name(var),
                'type': self._determine_variable_type(var),
                'description': description,
                'required': var.get('required', True)
            }
            
            # Add constraints based on type
            if variable_info['type'] == 'categorical':
                variable_info['options'] = self._get_categorical_options(var)
            elif variable_info['type'] in ['number', 'integer']:
                min_val = var.get('Minimum - for numerical', var.get('minimum', var.get('min')))
                max_val = var.get('Maximum - for numerical', var.get('maximum', var.get('max')))
                
                # Extract @value if it's a dict
                if isinstance(min_val, dict):
                    min_val = min_val.get('@value')
                if isinstance(max_val, dict):
                    max_val = max_val.get('@value')
                
                variable_info['min'] = min_val
                variable_info['max'] = max_val
            
            parsed.append(variable_info)
        
        return parsed
    
    def _parse_schema_properties(self, properties: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse variables from JSON Schema properties."""
        variables = []
        
        for prop_name, prop_def in properties.items():
            if not isinstance(prop_def, dict):
                continue
            
            variable_info = {
                'name': prop_name,
                'label': prop_def.get('title', prop_name.replace('_', ' ').title()),
                'type': self._schema_type_to_variable_type(prop_def.get('type', 'string')),
                'description': prop_def.get('description', ''),
                'required': True  # Can be refined based on schema 'required' array
            }
            
            if 'enum' in prop_def:
                variable_info['type'] = 'categorical'
                variable_info['options'] = prop_def['enum']
            elif variable_info['type'] in ['number', 'integer']:
                variable_info['min'] = prop_def.get('minimum')
                variable_info['max'] = prop_def.get('maximum')
            
            variables.append(variable_info)
        
        return variables
    
    def _get_human_readable_name(self, var: Dict[str, Any]) -> str:
        """Get a human-readable label for a variable."""
        # Try different fields for human-readable names
        for field in ['label', 'title', 'displayName', 'description']:
            if field in var and var[field]:
                return str(var[field])
        
        # Fall back to formatting the name field
        name = var.get('name', var.get('id', 'unknown'))
        return name.replace('_', ' ').replace('-', ' ').title()
    
    def _determine_variable_type(self, var: Dict[str, Any]) -> str:
        """Determine the variable type (categorical, number, integer, text)."""
        # Check for explicit type in FAIRmodels format
        type_of_input = var.get('Type of input', {})
        if isinstance(type_of_input, dict):
            type_of_input = type_of_input.get('@value', '')
        
        if type_of_input:
            type_of_input = str(type_of_input).lower()
            if type_of_input in ['categorical', 'enum', 'choice']:
                return 'categorical'
            elif type_of_input in ['number', 'float', 'double', 'numeric']:
                return 'number'
            elif type_of_input in ['integer', 'int']:
                return 'integer'
        
        # Check for explicit type
        var_type = var.get('type', '').lower()
        
        if var_type in ['categorical', 'enum', 'choice']:
            return 'categorical'
        elif var_type in ['number', 'float', 'double', 'numeric']:
            return 'number'
        elif var_type in ['integer', 'int']:
            return 'integer'
        elif var_type in ['boolean', 'bool']:
            return 'boolean'
        elif var_type in ['text', 'string']:
            return 'text'
        
        # Infer from presence of min/max (check this BEFORE categories)
        if 'Minimum - for numerical' in var or 'Maximum - for numerical' in var:
            # Check if at least one has a non-null value
            min_val = var.get('Minimum - for numerical', {})
            max_val = var.get('Maximum - for numerical', {})
            if isinstance(min_val, dict):
                min_val = min_val.get('@value')
            if isinstance(max_val, dict):
                max_val = max_val.get('@value')
            
            if min_val is not None or max_val is not None:
                return 'number'
        
        if 'minimum' in var or 'maximum' in var or 'min' in var or 'max' in var:
            return 'number'
        
        # Infer from presence of non-empty Categories
        if 'Categories' in var and isinstance(var['Categories'], list):
            # Check if Categories has valid entries (not just empty/null)
            valid_categories = False
            for cat in var['Categories']:
                if isinstance(cat, dict):
                    identification = cat.get('Identification for category used in model', {})
                    if isinstance(identification, dict):
                        identification = identification.get('@value')
                    if identification is not None:
                        valid_categories = True
                        break
            if valid_categories:
                return 'categorical'
        
        # Infer from presence of options/enum
        if 'options' in var or 'enum' in var or 'choices' in var:
            return 'categorical'
        
        # Default to text
        return 'text'
    
    def _schema_type_to_variable_type(self, schema_type: str) -> str:
        """Convert JSON Schema type to variable type."""
        mapping = {
            'number': 'number',
            'integer': 'integer',
            'boolean': 'boolean',
            'string': 'text'
        }
        return mapping.get(schema_type.lower(), 'text')
    
    def _get_categorical_options(self, var: Dict[str, Any]) -> List[Any]:
        """Extract categorical options from variable definition."""
        # Handle FAIRmodels.org Categories format
        if 'Categories' in var and isinstance(var['Categories'], list):
            options = []
            for category in var['Categories']:
                if not isinstance(category, dict):
                    continue
                
                # Get the identification value (what the model expects)
                identification = category.get('Identification for category used in model', {})
                if isinstance(identification, dict):
                    identification = identification.get('@value', '')
                
                # Get the category label (human-readable)
                category_label = category.get('Category Label', {})
                if isinstance(category_label, dict):
                    rdfs_label = category_label.get('rdfs:label', {})
                    if isinstance(rdfs_label, dict):
                        label = rdfs_label.get('@value', '')
                    else:
                        label = rdfs_label
                else:
                    label = ''
                
                # Use identification value for the option, with label as display
                if identification:
                    # Store as dict with both value and label for better display
                    options.append({
                        'value': identification,
                        'label': label if label else identification
                    })
            
            if options:
                return options
        
        # Try different possible fields
        for field in ['options', 'enum', 'choices', 'values']:
            if field in var and isinstance(var[field], list):
                return var[field]
        
        return []
