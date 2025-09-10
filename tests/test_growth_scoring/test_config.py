"""
Tests for growth rate configuration management.

This module tests the configuration dataclasses and loading functionality
for the growth rate scoring system.
"""

import pytest
import tempfile
from pathlib import Path
import yaml
from typing import Dict, Any

# Import the modules we'll implement
from antigentools.scoring.config import (
    GrowthRateConfig,
    ConvergenceConfig,
    load_config,
    parse_growth_rate_config
)


class TestGrowthRateConfig:
    """Test the GrowthRateConfig dataclass."""
    
    def test_default_values(self):
        """Test that default values are correctly set."""
        config = GrowthRateConfig()
        
        assert config.connect_gaps is True
        assert config.min_segment_length == 3
        assert config.min_sequence_count == 10
        assert config.min_variant_frequency == 0.01
        assert config.epsilon == 1e-3
        assert config.min_total_sequences == 300
        assert config.spline_smoothing_factor == 1.0
        assert config.spline_order == 3
    
    def test_custom_values(self):
        """Test creating config with custom values."""
        config = GrowthRateConfig(
            connect_gaps=False,
            min_segment_length=5,
            min_sequence_count=20,
            min_variant_frequency=0.05,
            epsilon=1e-4,
            min_total_sequences=500
        )
        
        assert config.connect_gaps is False
        assert config.min_segment_length == 5
        assert config.min_sequence_count == 20
        assert config.min_variant_frequency == 0.05
        assert config.epsilon == 1e-4
        assert config.min_total_sequences == 500
    
    def test_validation_min_segment_length(self):
        """Test validation of min_segment_length."""
        with pytest.raises(ValueError, match="min_segment_length must be >= 1"):
            GrowthRateConfig(min_segment_length=0)
    
    def test_validation_min_sequence_count(self):
        """Test validation of min_sequence_count."""
        with pytest.raises(ValueError, match="min_sequence_count must be >= 0"):
            GrowthRateConfig(min_sequence_count=-1)
    
    def test_validation_min_variant_frequency(self):
        """Test validation of min_variant_frequency."""
        with pytest.raises(ValueError, match="min_variant_frequency must be between 0 and 1"):
            GrowthRateConfig(min_variant_frequency=-0.1)
        
        with pytest.raises(ValueError, match="min_variant_frequency must be between 0 and 1"):
            GrowthRateConfig(min_variant_frequency=1.5)
    
    def test_validation_epsilon(self):
        """Test validation of epsilon."""
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            GrowthRateConfig(epsilon=0)
        
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            GrowthRateConfig(epsilon=-0.001)
    
    def test_validation_min_total_sequences_optional(self):
        """Test that min_total_sequences can be None."""
        config = GrowthRateConfig(min_total_sequences=None)
        assert config.min_total_sequences is None
    
    def test_validation_spline_order(self):
        """Test validation of spline_order."""
        with pytest.raises(ValueError, match="spline_order must be between 1 and 5"):
            GrowthRateConfig(spline_order=0)
        
        with pytest.raises(ValueError, match="spline_order must be between 1 and 5"):
            GrowthRateConfig(spline_order=6)


class TestConvergenceConfig:
    """Test the ConvergenceConfig dataclass."""
    
    def test_default_values(self):
        """Test that default values are correctly set."""
        config = ConvergenceConfig()
        
        assert config.threshold == 0.5
        assert config.check_diagnostics is True
        assert config.required_models == ["FGA", "GARW"]
    
    def test_custom_values(self):
        """Test creating config with custom values."""
        config = ConvergenceConfig(
            threshold=0.3,
            check_diagnostics=False,
            required_models=["FGA"]
        )
        
        assert config.threshold == 0.3
        assert config.check_diagnostics is False
        assert config.required_models == ["FGA"]
    
    def test_validation_threshold(self):
        """Test validation of convergence threshold."""
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            ConvergenceConfig(threshold=-0.1)
        
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            ConvergenceConfig(threshold=1.5)
    
    def test_validation_required_models(self):
        """Test validation of required models list."""
        # Empty list should be allowed
        config = ConvergenceConfig(required_models=[])
        assert config.required_models == []
        
        # Non-list should raise error
        with pytest.raises(TypeError):
            ConvergenceConfig(required_models="FGA")


class TestConfigLoading:
    """Test configuration loading from YAML files."""
    
    def test_load_config_from_yaml(self):
        """Test loading configuration from a YAML file."""
        yaml_content = """
        growth_rate:
          connect_gaps: false
          min_segment_length: 5
          min_sequence_count: 20
          min_variant_frequency: 0.05
          epsilon: 0.001
          min_total_sequences: 500
          spline_smoothing_factor: 2.0
          spline_order: 4
        
        convergence:
          threshold: 0.3
          check_diagnostics: false
          required_models:
            - FGA
            - GARW
            - MLR
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)
        
        try:
            config_dict = load_config(temp_path)
            
            assert 'growth_rate' in config_dict
            assert 'convergence' in config_dict
            assert config_dict['growth_rate']['connect_gaps'] is False
            assert config_dict['growth_rate']['min_segment_length'] == 5
            assert config_dict['convergence']['threshold'] == 0.3
            assert len(config_dict['convergence']['required_models']) == 3
        finally:
            temp_path.unlink()
    
    def test_load_config_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("non_existent_file.yaml"))
    
    def test_parse_config(self):
        """Test parsing configuration dictionary into dataclasses."""
        config_dict = {
            'growth_rate': {
                'connect_gaps': False,
                'min_segment_length': 5,
                'min_sequence_count': 20,
                'min_variant_frequency': 0.05,
                'epsilon': 0.001,
                'min_total_sequences': 500
            },
            'convergence': {
                'threshold': 0.3,
                'check_diagnostics': False,
                'required_models': ['FGA', 'GARW']
            }
        }
        
        growth_config, conv_config = parse_growth_rate_config(config_dict)
        
        # Check growth rate config
        assert isinstance(growth_config, GrowthRateConfig)
        assert growth_config.connect_gaps is False
        assert growth_config.min_segment_length == 5
        assert growth_config.min_sequence_count == 20
        assert growth_config.min_variant_frequency == 0.05
        assert growth_config.epsilon == 0.001
        assert growth_config.min_total_sequences == 500
        
        # Check convergence config
        assert isinstance(conv_config, ConvergenceConfig)
        assert conv_config.threshold == 0.3
        assert conv_config.check_diagnostics is False
        assert conv_config.required_models == ['FGA', 'GARW']
    
    def test_parse_config_with_defaults(self):
        """Test parsing with missing sections uses defaults."""
        # Empty config
        growth_config, conv_config = parse_growth_rate_config({})
        
        assert isinstance(growth_config, GrowthRateConfig)
        assert isinstance(conv_config, ConvergenceConfig)
        assert growth_config.min_segment_length == 3  # default
        assert conv_config.threshold == 0.5  # default
        
        # Only growth_rate section
        config_dict = {
            'growth_rate': {
                'min_segment_length': 5
            }
        }
        growth_config, conv_config = parse_growth_rate_config(config_dict)
        
        assert growth_config.min_segment_length == 5
        assert growth_config.min_sequence_count == 10  # default
        assert conv_config.threshold == 0.5  # default
    
    def test_parse_config_ignores_extra_fields(self):
        """Test that extra fields in config are ignored."""
        config_dict = {
            'growth_rate': {
                'min_segment_length': 5,
                'extra_field': 'ignored'
            },
            'convergence': {
                'threshold': 0.3,
                'another_extra': 42
            },
            'completely_extra_section': {
                'data': 'ignored'
            }
        }
        
        growth_config, conv_config = parse_growth_rate_config(config_dict)
        
        assert growth_config.min_segment_length == 5
        assert conv_config.threshold == 0.3
        # Should not raise any errors for extra fields
    
    def test_parse_config_validation_errors(self):
        """Test that validation errors are properly raised."""
        config_dict = {
            'growth_rate': {
                'min_segment_length': -1  # Invalid
            }
        }
        
        with pytest.raises(ValueError, match="min_segment_length must be >= 1"):
            parse_growth_rate_config(config_dict)
    
    def test_config_to_dict(self):
        """Test converting configs back to dictionary."""
        growth_config = GrowthRateConfig(
            connect_gaps=False,
            min_segment_length=5
        )
        
        # Configs should have a to_dict method for saving
        config_dict = growth_config.to_dict()
        
        assert config_dict['connect_gaps'] is False
        assert config_dict['min_segment_length'] == 5
        assert config_dict['min_sequence_count'] == 10  # default
    
    def test_round_trip_yaml(self):
        """Test saving and loading configuration round-trip."""
        growth_config = GrowthRateConfig(
            connect_gaps=False,
            min_segment_length=5,
            epsilon=0.002
        )
        conv_config = ConvergenceConfig(
            threshold=0.4,
            required_models=["FGA"]
        )
        
        # Create config dict
        full_config = {
            'growth_rate': growth_config.to_dict(),
            'convergence': conv_config.to_dict()
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(full_config, f)
            temp_path = Path(f.name)
        
        try:
            # Load and parse
            loaded_dict = load_config(temp_path)
            loaded_growth, loaded_conv = parse_growth_rate_config(loaded_dict)
            
            # Check values match
            assert loaded_growth.connect_gaps == growth_config.connect_gaps
            assert loaded_growth.min_segment_length == growth_config.min_segment_length
            assert loaded_growth.epsilon == growth_config.epsilon
            assert loaded_conv.threshold == conv_config.threshold
            assert loaded_conv.required_models == conv_config.required_models
        finally:
            temp_path.unlink()