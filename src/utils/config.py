"""
Configuration management for MOSDAC AI Help Bot.
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class Config:
    """Configuration manager that loads and provides access to settings."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. Defaults to config/settings.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            return config or {}
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'app.name')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        env_mappings = {
            'MOSDAC_DEBUG': 'app.debug',
            'MOSDAC_LOG_LEVEL': 'app.log_level',
            'MOSDAC_API_HOST': 'api.host',
            'MOSDAC_API_PORT': 'api.port',
            'MOSDAC_DB_PATH': 'database.path',
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if env_value.lower() in ('true', 'false'):
                    env_value = env_value.lower() == 'true'
                elif env_value.isdigit():
                    env_value = int(env_value)
                
                self.set(config_key, env_value)
    
    @property
    def app_name(self) -> str:
        """Get application name."""
        return self.get('app.name', 'MOSDAC AI Help Bot')
    
    @property
    def debug(self) -> bool:
        """Get debug mode setting."""
        return self.get('app.debug', False)
    
    @property
    def portal_base_url(self) -> str:
        """Get portal base URL."""
        return self.get('portal.base_url', 'https://www.mosdac.gov.in')
    
    @property
    def vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration."""
        return self.get('vector_db', {})
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get('models', {})
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration. Uses original path if None.
        """
        save_path = Path(output_path) if output_path else self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(self._config, file, default_flow_style=False, sort_keys=False)

# Global configuration instance
config = Config()
config.update_from_env()