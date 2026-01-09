"""
Unit tests for configuration management.
"""

import pytest
import tempfile
import os
from pathlib import Path
from geosuite.config import ConfigManager, load_config, get_config, set_config


class TestConfigManager:
    """Tests for ConfigManager class."""
    
    def test_init_empty(self):
        """Test initialization with empty config."""
        config = ConfigManager()
        assert config.to_dict() == {}
    
    def test_init_with_config(self):
        """Test initialization with initial config."""
        initial = {'petro': {'archie': {'a': 1.0}}}
        config = ConfigManager(initial)
        assert config.get('petro.archie.a') == 1.0
    
    def test_get_simple_key(self):
        """Test getting simple key."""
        config = ConfigManager({'key': 'value'})
        assert config.get('key') == 'value'
    
    def test_get_nested_key(self):
        """Test getting nested key with dot notation."""
        config = ConfigManager({
            'petro': {
                'archie': {
                    'a': 1.0
                }
            }
        })
        assert config.get('petro.archie.a') == 1.0
    
    def test_get_default(self):
        """Test getting default value for missing key."""
        config = ConfigManager()
        assert config.get('missing.key', default='default') == 'default'
    
    def test_set_simple_key(self):
        """Test setting simple key."""
        config = ConfigManager()
        config.set('key', 'value')
        assert config.get('key') == 'value'
    
    def test_set_nested_key(self):
        """Test setting nested key with dot notation."""
        config = ConfigManager()
        config.set('petro.archie.a', 1.0)
        assert config.get('petro.archie.a') == 1.0
    
    def test_load_yaml(self):
        """Test loading YAML configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("petro:\n  archie:\n    a: 1.0\n")
            f.flush()
            temp_path = f.name
        
        # File is closed now, safe to delete on Windows
        try:
            config = ConfigManager()
            config.load_from_file(temp_path)
            assert config.get('petro.archie.a') == 1.0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_load_json(self):
        """Test loading JSON configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"petro": {"archie": {"a": 1.0}}}')
            f.flush()
            temp_path = f.name
        
        # File is closed now, safe to delete on Windows
        try:
            config = ConfigManager()
            config.load_from_file(temp_path)
            assert config.get('petro.archie.a') == 1.0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_load_file_not_found(self):
        """Test error when file doesn't exist."""
        config = ConfigManager()
        with pytest.raises(FileNotFoundError):
            config.load_from_file('nonexistent.yaml')
    
    def test_load_unsupported_format(self):
        """Test error with unsupported file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test")
            f.flush()
            temp_path = f.name
        
        # File is closed now, safe to delete on Windows
        try:
            config = ConfigManager()
            with pytest.raises(ValueError, match="Unsupported file format"):
                config.load_from_file(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_merge_config(self):
        """Test merging configurations."""
        base = {'petro': {'archie': {'a': 1.0, 'm': 2.0}}}
        update = {'petro': {'archie': {'a': 1.5}}}
        
        config = ConfigManager(base)
        config._config = config._merge_config(base, update)
        
        assert config.get('petro.archie.a') == 1.5
        assert config.get('petro.archie.m') == 2.0
    
    def test_save_yaml(self):
        """Test saving YAML configuration."""
        config = ConfigManager({'key': 'value'})
        
        with tempfile.NamedTemporaryFile(mode='r', suffix='.yaml', delete=False) as f:
            f.close()
            config.save_to_file(f.name, format='yaml')
            
            loaded = ConfigManager()
            loaded.load_from_file(f.name)
            assert loaded.get('key') == 'value'
            
            os.unlink(f.name)
    
    def test_save_json(self):
        """Test saving JSON configuration."""
        config = ConfigManager({'key': 'value'})
        
        with tempfile.NamedTemporaryFile(mode='r', suffix='.json', delete=False) as f:
            f.close()
            config.save_to_file(f.name, format='json')
            
            loaded = ConfigManager()
            loaded.load_from_file(f.name)
            assert loaded.get('key') == 'value'
            
            os.unlink(f.name)


class TestGlobalConfig:
    """Tests for global configuration functions."""
    
    def test_load_config(self):
        """Test loading global configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("key: value\n")
            f.flush()
            temp_path = f.name
        
        # File is closed now, safe to delete on Windows
        try:
            config = load_config(temp_path)
            assert isinstance(config, ConfigManager)
            assert config.get('key') == 'value'
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_get_config(self):
        """Test getting from global configuration."""
        set_config('test.key', 'value')
        assert get_config('test.key') == 'value'
    
    def test_set_config(self):
        """Test setting global configuration."""
        set_config('test.key2', 'value2')
        assert get_config('test.key2') == 'value2'

