import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class InputValidator:
    """Enhanced input validation for API prediction requests"""
    
    def __init__(self):
        self.valid_methods = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS']
        self.trusted_domains = [
            'raw.githubusercontent.com',
            'api.github.com', 
            'petstore.swagger.io',
            'httpbin.org',
            'jsonplaceholder.typicode.com'
        ]
        
    def validate_request(self, request_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Comprehensive request validation
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate user_id
        if not self._validate_user_id(request_data.get('user_id')):
            errors.append("user_id must be a non-empty string (max 100 chars)")
        
        # Validate events
        events_errors = self._validate_events(request_data.get('events', []))
        errors.extend(events_errors)
        
        # Validate prompt
        if not self._validate_prompt(request_data.get('prompt')):
            errors.append("prompt must be a string (max 500 chars) if provided")
        
        # Validate spec_url
        if not self._validate_spec_url(request_data.get('spec_url')):
            errors.append("spec_url must be a valid HTTP/HTTPS URL")
        
        # Validate k
        if not self._validate_k(request_data.get('k')):
            errors.append("k must be an integer between 1 and 20")
        
        return len(errors) == 0, errors
    
    def _validate_user_id(self, user_id: Any) -> bool:
        """Validate user_id field"""
        if not isinstance(user_id, str):
            return False
        if len(user_id.strip()) == 0 or len(user_id) > 100:
            return False
        # Check for valid characters (alphanumeric, hyphens, underscores)
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            return False
        return True
    
    def _validate_events(self, events: Any) -> List[str]:
        """Validate events array"""
        errors = []
        
        if not isinstance(events, list):
            errors.append("events must be an array")
            return errors
        
        if len(events) == 0:
            errors.append("at least one event is required")
            return errors
        
        if len(events) > 50:
            errors.append("maximum 50 events allowed")
        
        for i, event in enumerate(events):
            event_errors = self._validate_single_event(event, i)
            errors.extend(event_errors)
        
        # Validate event chronology
        chronology_errors = self._validate_event_chronology(events)
        errors.extend(chronology_errors)
        
        return errors
    
    def _validate_single_event(self, event: Any, index: int) -> List[str]:
        """Validate a single event"""
        errors = []
        prefix = f"events[{index}]"
        
        if not isinstance(event, dict):
            errors.append(f"{prefix} must be an object")
            return errors
        
        # Validate timestamp
        ts = event.get('ts')
        if not isinstance(ts, str):
            errors.append(f"{prefix}.ts must be a string")
        else:
            try:
                # Try to parse ISO format
                datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except ValueError:
                errors.append(f"{prefix}.ts must be a valid ISO timestamp")
        
        # Validate endpoint
        endpoint = event.get('endpoint')
        if not isinstance(endpoint, str):
            errors.append(f"{prefix}.endpoint must be a string")
        else:
            endpoint_errors = self._validate_endpoint_format(endpoint, prefix)
            errors.extend(endpoint_errors)
        
        # Validate params
        params = event.get('params')
        if params is not None:
            if not isinstance(params, dict):
                errors.append(f"{prefix}.params must be an object if provided")
            elif len(json.dumps(params)) > 10000:  # Limit param size
                errors.append(f"{prefix}.params too large (max 10KB)")
        
        return errors
    
    def _validate_endpoint_format(self, endpoint: str, prefix: str) -> List[str]:
        """Validate endpoint format"""
        errors = []
        
        if len(endpoint.strip()) == 0:
            errors.append(f"{prefix}.endpoint cannot be empty")
            return errors
        
        if len(endpoint) > 500:
            errors.append(f"{prefix}.endpoint too long (max 500 chars)")
        
        # Check format: "METHOD /path"
        parts = endpoint.strip().split(' ', 1)
        if len(parts) != 2:
            errors.append(f"{prefix}.endpoint must be in format 'METHOD /path'")
            return errors
        
        method, path = parts
        
        # Validate method
        if method.upper() not in self.valid_methods:
            errors.append(f"{prefix}.endpoint has invalid method '{method}'")
        
        # Validate path
        if not path.startswith('/'):
            errors.append(f"{prefix}.endpoint path must start with '/'")
        
        # Check for suspicious patterns
        if any(pattern in path.lower() for pattern in ['../', 'javascript:', 'data:']):
            errors.append(f"{prefix}.endpoint path contains suspicious patterns")
        
        return errors
    
    def _validate_event_chronology(self, events: List[Dict]) -> List[str]:
        """Validate events are in chronological order"""
        errors = []
        
        if len(events) <= 1:
            return errors
        
        try:
            timestamps = []
            for event in events:
                if isinstance(event.get('ts'), str):
                    ts = datetime.fromisoformat(event['ts'].replace('Z', '+00:00'))
                    timestamps.append(ts)
            
            # Check if timestamps are reasonably ordered (allow some tolerance)
            for i in range(1, len(timestamps)):
                if timestamps[i] < timestamps[i-1]:
                    # Allow small differences (up to 1 second) for concurrent requests
                    diff = (timestamps[i-1] - timestamps[i]).total_seconds()
                    if diff > 1.0:
                        errors.append("events should be in chronological order")
                        break
        except Exception as e:
            logger.debug(f"Could not validate chronology: {e}")
            # Don't fail validation for chronology issues
        
        return errors
    
    def _validate_prompt(self, prompt: Any) -> bool:
        """Validate prompt field"""
        if prompt is None:
            return True  # Optional field
        
        if not isinstance(prompt, str):
            return False
        
        if len(prompt) > 500:
            return False
        
        # Check for suspicious content
        suspicious_patterns = [
            'javascript:', 'data:', 'vbscript:', '<script',
            'eval(', 'exec(', 'system(', 'shell_exec'
        ]
        prompt_lower = prompt.lower()
        if any(pattern in prompt_lower for pattern in suspicious_patterns):
            return False
        
        return True
    
    def _validate_spec_url(self, spec_url: Any) -> bool:
        """Validate OpenAPI spec URL"""
        if not isinstance(spec_url, str):
            return False
        
        if len(spec_url) > 2000:
            return False
        
        try:
            parsed = urlparse(spec_url)
            
            # Must be HTTP/HTTPS
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Must have a valid hostname
            if not parsed.netloc:
                return False
            
            # Security check: warn about non-trusted domains
            domain = parsed.netloc.lower()
            if not any(trusted in domain for trusted in self.trusted_domains):
                logger.warning(f"Using non-trusted domain: {domain}")
                # Don't fail validation, just log warning
            
            return True
            
        except Exception:
            return False
    
    def _validate_k(self, k: Any) -> bool:
        """Validate k parameter"""
        if not isinstance(k, int):
            return False
        
        return 1 <= k <= 20


class ParameterGenerator:
    """Generate realistic parameters for API endpoints based on OpenAPI schemas"""
    
    def __init__(self):
        self.sample_values = {
            'string': ['test', 'example', 'sample'],
            'integer': [1, 42, 100],
            'number': [1.0, 3.14, 42.5],
            'boolean': [True, False],
            'array': [['item1', 'item2'], [1, 2, 3]],
            'object': [{'key': 'value'}]
        }
        
        # Common parameter patterns
        self.parameter_patterns = {
            'id': {'type': 'string', 'examples': ['123', 'abc-123', 'user_456']},
            'user_id': {'type': 'string', 'examples': ['user_123', 'u-456', 'usr_789']},
            'email': {'type': 'string', 'examples': ['test@example.com', 'user@domain.org']},
            'name': {'type': 'string', 'examples': ['John Doe', 'Test User', 'Sample Name']},
            'status': {'type': 'string', 'examples': ['active', 'pending', 'completed']},
            'limit': {'type': 'integer', 'examples': [10, 25, 50]},
            'offset': {'type': 'integer', 'examples': [0, 10, 20]},
            'page': {'type': 'integer', 'examples': [1, 2, 3]},
            'amount': {'type': 'number', 'examples': [10.99, 25.50, 100.00]},
            'quantity': {'type': 'integer', 'examples': [1, 2, 5]},
            'date': {'type': 'string', 'examples': ['2024-01-01', '2024-12-31']},
        }
    
    def generate_parameters(
        self, 
        endpoint_data: Dict[str, Any], 
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate realistic parameters for an endpoint
        
        Args:
            endpoint_data: OpenAPI endpoint information
            user_context: Context from recent user events
        
        Returns:
            Generated parameters
        """
        params = {}
        
        # Generate path parameters
        path_params = self._generate_path_parameters(endpoint_data, user_context)
        params.update(path_params)
        
        # Generate query parameters
        query_params = self._generate_query_parameters(endpoint_data, user_context)
        params.update(query_params)
        
        # Generate request body (for POST/PUT/PATCH)
        body_params = self._generate_body_parameters(endpoint_data, user_context)
        if body_params:
            params.update(body_params)
        
        return params
    
    def _generate_path_parameters(
        self, 
        endpoint_data: Dict[str, Any], 
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate path parameters (e.g., /users/{id})"""
        params = {}
        
        endpoint = endpoint_data.get('endpoint', '')
        path = endpoint.split(' ', 1)[1] if ' ' in endpoint else ''
        
        # Find path parameters (enclosed in {})
        path_param_pattern = r'\{([^}]+)\}'
        path_params = re.findall(path_param_pattern, path)
        
        for param_name in path_params:
            value = self._generate_parameter_value(param_name, 'path', user_context)
            params[param_name] = value
        
        return params
    
    def _generate_query_parameters(
        self, 
        endpoint_data: Dict[str, Any], 
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate query parameters"""
        params = {}
        
        # Get parameters from OpenAPI spec
        api_parameters = endpoint_data.get('parameters', [])
        query_params = [p for p in api_parameters if p.get('in') == 'query']
        
        for param in query_params:
            param_name = param.get('name')
            if param_name and param.get('required', False):
                value = self._generate_parameter_value(
                    param_name, 
                    'query', 
                    user_context, 
                    param_schema=param.get('schema', {})
                )
                params[param_name] = value
        
        # Add common query parameters for GET requests
        method = endpoint_data.get('endpoint', '').split()[0]
        if method == 'GET':
            # Add pagination if not already specified
            if not any(p.get('name') in ['limit', 'page', 'per_page'] for p in query_params):
                if 'list' in endpoint_data.get('endpoint', '').lower():
                    params['limit'] = 10
        
        return params
    
    def _generate_body_parameters(
        self, 
        endpoint_data: Dict[str, Any], 
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate request body parameters"""
        method = endpoint_data.get('endpoint', '').split()[0]
        
        if method not in ['POST', 'PUT', 'PATCH']:
            return {}
        
        request_body = endpoint_data.get('request_body')
        if not request_body:
            return {}
        
        schema = request_body.get('schema', {})
        return self._generate_from_schema(schema, user_context)
    
    def _generate_parameter_value(
        self, 
        param_name: str, 
        param_location: str, 
        user_context: Optional[Dict[str, Any]], 
        param_schema: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Generate a value for a specific parameter"""
        
        # Try to use context from user's recent events
        if user_context and param_name in user_context:
            return user_context[param_name]
        
        # Use parameter patterns
        param_name_lower = param_name.lower()
        for pattern, config in self.parameter_patterns.items():
            if pattern in param_name_lower:
                return self._pick_random_example(config['examples'])
        
        # Use schema if provided
        if param_schema:
            return self._generate_from_schema(param_schema, user_context)
        
        # Default based on parameter name patterns
        if 'id' in param_name_lower:
            return '123'
        elif 'name' in param_name_lower:
            return 'example'
        elif 'email' in param_name_lower:
            return 'test@example.com'
        elif 'count' in param_name_lower or 'limit' in param_name_lower:
            return 10
        else:
            return 'example'
    
    def _generate_from_schema(
        self, 
        schema: Dict[str, Any], 
        user_context: Optional[Dict[str, Any]]
    ) -> Any:
        """Generate value from OpenAPI schema"""
        
        schema_type = schema.get('type', 'string')
        
        # Handle examples from schema
        if 'example' in schema:
            return schema['example']
        
        if 'examples' in schema and schema['examples']:
            return self._pick_random_example(schema['examples'])
        
        # Handle different types
        if schema_type == 'string':
            return self._generate_string_value(schema)
        elif schema_type == 'integer':
            return self._generate_integer_value(schema)
        elif schema_type == 'number':
            return self._generate_number_value(schema)
        elif schema_type == 'boolean':
            return True
        elif schema_type == 'array':
            items_schema = schema.get('items', {})
            return [self._generate_from_schema(items_schema, user_context)]
        elif schema_type == 'object':
            return self._generate_object_value(schema, user_context)
        else:
            return 'example'
    
    def _generate_string_value(self, schema: Dict[str, Any]) -> str:
        """Generate string value based on schema"""
        # Check for format
        format_type = schema.get('format')
        if format_type == 'email':
            return 'test@example.com'
        elif format_type == 'date':
            return '2024-01-01'
        elif format_type == 'date-time':
            return '2024-01-01T12:00:00Z'
        elif format_type == 'uuid':
            return '123e4567-e89b-12d3-a456-426614174000'
        
        # Check for enum values
        if 'enum' in schema:
            return self._pick_random_example(schema['enum'])
        
        # Check length constraints
        min_length = schema.get('minLength', 1)
        max_length = schema.get('maxLength', 50)
        
        if min_length <= 7 <= max_length:
            return 'example'
        elif min_length > 7:
            return 'example' + 'x' * (min_length - 7)
        else:
            return 'example'[:max_length]
    
    def _generate_integer_value(self, schema: Dict[str, Any]) -> int:
        """Generate integer value based on schema"""
        minimum = schema.get('minimum', 1)
        maximum = schema.get('maximum', 100)
        
        # Prefer common values within range
        common_values = [1, 10, 42, 100]
        for value in common_values:
            if minimum <= value <= maximum:
                return value
        
        return max(minimum, min(maximum, 42))
    
    def _generate_number_value(self, schema: Dict[str, Any]) -> float:
        """Generate number value based on schema"""
        minimum = schema.get('minimum', 0.0)
        maximum = schema.get('maximum', 100.0)
        
        return max(minimum, min(maximum, 42.0))
    
    def _generate_object_value(
        self, 
        schema: Dict[str, Any], 
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate object value based on schema"""
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        
        obj = {}
        
        # Generate required properties
        for prop_name in required:
            if prop_name in properties:
                prop_schema = properties[prop_name]
                obj[prop_name] = self._generate_from_schema(prop_schema, user_context)
        
        # Generate some optional properties (up to 3)
        optional_props = [p for p in properties.keys() if p not in required]
        for prop_name in optional_props[:3]:
            prop_schema = properties[prop_name]
            obj[prop_name] = self._generate_from_schema(prop_schema, user_context)
        
        return obj
    
    def _pick_random_example(self, examples: List[Any]) -> Any:
        """Pick a random example from a list"""
        if not examples:
            return 'example'
        return examples[0]  # For consistency, always pick first example
    
    def extract_user_context(self, events: List[Any]) -> Dict[str, Any]:
        """Extract context values from user's recent events"""
        context = {}
        
        for event in events:
            # Extract IDs from paths
            endpoint = event.endpoint
            path = endpoint.split(' ', 1)[1] if ' ' in endpoint else ''
            
            # Look for ID patterns in path
            id_pattern = r'/([a-zA-Z0-9_-]+)'
            matches = re.findall(id_pattern, path)
            
            # Store potential ID values
            for match in matches:
                if len(match) > 2:  # Likely an ID
                    if 'user' in path.lower():
                        context['user_id'] = match
                    elif 'product' in path.lower():
                        context['product_id'] = match
                    elif 'order' in path.lower():
                        context['order_id'] = match
                    else:
                        context['id'] = match
            
            # Extract parameters from event
            params = getattr(event, 'params', {})
            if isinstance(params, dict):
                for key, value in params.items():
                    if key not in context and value:
                        context[key] = value
        
        return context