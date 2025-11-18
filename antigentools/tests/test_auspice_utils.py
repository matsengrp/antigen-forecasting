"""Unit tests for auspice-related utility functions."""

import pytest
import json
import tempfile
import os
from antigentools.utils import extract_clade_assignments_from_auspice


class TestExtractCladeAssignmentsFromAuspice:
    """Test extract_clade_assignments_from_auspice function."""

    def test_basic_tree_extraction(self):
        """Test extracting clade assignments from a basic tree structure."""
        # Create a mock auspice JSON with simple tree
        auspice_data = {
            "tree": {
                "name": "NODE_0",
                "node_attrs": {
                    "clade_membership": {"value": 1}
                },
                "children": [
                    {
                        "name": "tip1",
                        "node_attrs": {
                            "clade_membership": {"value": 1}
                        }
                    },
                    {
                        "name": "tip2",
                        "node_attrs": {
                            "clade_membership": {"value": 2}
                        }
                    }
                ]
            }
        }

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(auspice_data, f)
            temp_path = f.name

        try:
            result = extract_clade_assignments_from_auspice(temp_path)

            # Check results
            assert len(result) == 2
            assert result['tip1'] == 1
            assert result['tip2'] == 2
        finally:
            os.unlink(temp_path)

    def test_nested_tree_extraction(self):
        """Test extracting from nested tree with multiple levels."""
        auspice_data = {
            "tree": {
                "name": "root",
                "node_attrs": {
                    "clade_membership": {"value": 0}
                },
                "children": [
                    {
                        "name": "internal1",
                        "node_attrs": {
                            "clade_membership": {"value": 1}
                        },
                        "children": [
                            {
                                "name": "tip1",
                                "node_attrs": {
                                    "clade_membership": {"value": 1}
                                }
                            },
                            {
                                "name": "tip2",
                                "node_attrs": {
                                    "clade_membership": {"value": 1}
                                }
                            }
                        ]
                    },
                    {
                        "name": "tip3",
                        "node_attrs": {
                            "clade_membership": {"value": 2}
                        }
                    }
                ]
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(auspice_data, f)
            temp_path = f.name

        try:
            result = extract_clade_assignments_from_auspice(temp_path)

            # Should only get tips, not internal nodes
            assert len(result) == 3
            assert result['tip1'] == 1
            assert result['tip2'] == 1
            assert result['tip3'] == 2
            assert 'internal1' not in result
            assert 'root' not in result
        finally:
            os.unlink(temp_path)

    def test_missing_clade_membership(self):
        """Test handling tips without clade_membership."""
        auspice_data = {
            "tree": {
                "name": "root",
                "node_attrs": {},
                "children": [
                    {
                        "name": "tip1",
                        "node_attrs": {
                            "clade_membership": {"value": 1}
                        }
                    },
                    {
                        "name": "tip2",
                        "node_attrs": {}  # No clade_membership
                    }
                ]
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(auspice_data, f)
            temp_path = f.name

        try:
            result = extract_clade_assignments_from_auspice(temp_path)

            # Should only include tip with valid clade_membership
            assert len(result) == 1
            assert result['tip1'] == 1
            assert 'tip2' not in result
        finally:
            os.unlink(temp_path)

    def test_missing_name(self):
        """Test handling nodes without name field."""
        auspice_data = {
            "tree": {
                "name": "root",
                "node_attrs": {},
                "children": [
                    {
                        "node_attrs": {  # Missing name
                            "clade_membership": {"value": 1}
                        }
                    },
                    {
                        "name": "tip1",
                        "node_attrs": {
                            "clade_membership": {"value": 2}
                        }
                    }
                ]
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(auspice_data, f)
            temp_path = f.name

        try:
            result = extract_clade_assignments_from_auspice(temp_path)

            # Should only include tip with valid name
            assert len(result) == 1
            assert result['tip1'] == 2
        finally:
            os.unlink(temp_path)

    def test_empty_tree(self):
        """Test handling tree with no tips."""
        auspice_data = {
            "tree": {
                "name": "root",
                "node_attrs": {
                    "clade_membership": {"value": 1}
                },
                "children": []
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(auspice_data, f)
            temp_path = f.name

        try:
            result = extract_clade_assignments_from_auspice(temp_path)

            # Empty result - root is not a tip
            assert len(result) == 0
        finally:
            os.unlink(temp_path)

    def test_invalid_json_path(self):
        """Test handling invalid file path."""
        with pytest.raises(FileNotFoundError):
            extract_clade_assignments_from_auspice("/nonexistent/path.json")

    def test_malformed_json(self):
        """Test handling malformed JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{invalid json")
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                extract_clade_assignments_from_auspice(temp_path)
        finally:
            os.unlink(temp_path)

    def test_integer_clade_values(self):
        """Test that clade membership values are correctly extracted as integers."""
        auspice_data = {
            "tree": {
                "name": "root",
                "children": [
                    {
                        "name": "tip1",
                        "node_attrs": {
                            "clade_membership": {"value": 0}
                        }
                    },
                    {
                        "name": "tip2",
                        "node_attrs": {
                            "clade_membership": {"value": 100}
                        }
                    }
                ]
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(auspice_data, f)
            temp_path = f.name

        try:
            result = extract_clade_assignments_from_auspice(temp_path)

            assert result['tip1'] == 0
            assert result['tip2'] == 100
            assert isinstance(result['tip1'], int)
            assert isinstance(result['tip2'], int)
        finally:
            os.unlink(temp_path)
