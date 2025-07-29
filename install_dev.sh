#!/bin/bash
# Development setup script for antigen-forecasting project
# This script sets up the environment so that antigentools imports work seamlessly in notebooks

echo "=== Antigen Forecasting Development Setup ==="
echo ""

# Check if mamba is available
if ! command -v mamba &> /dev/null; then
    echo "❌ Error: mamba is not installed or not in PATH"
    echo "Please install mamba first"
    exit 1
fi

echo "🔧 Setting up mamba environment..."

# Check if environment already exists
if mamba env list | grep -q "^antigen "; then
    echo "📝 Environment 'antigen' already exists. Updating..."
    mamba env update -f environment.yaml
else
    echo "🆕 Creating new environment 'antigen'..."
    mamba env create -f environment.yaml
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To use the environment:"
echo "  mamba activate antigen"
echo ""
echo "After activation, you can import antigentools in any notebook:"
echo "  from antigentools.antigen_reader import AntigenReader"
echo "  from antigentools.utils import hamming_distance, translate_dna_to_aa"
echo "  from antigentools.plot import plot_tree"