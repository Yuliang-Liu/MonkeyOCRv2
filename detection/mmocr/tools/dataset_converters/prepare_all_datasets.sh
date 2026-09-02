#!/bin/bash
# Copyright (c) OpenMMLab. All rights reserved.
# Script to prepare icdar2015, ctw1500, and totaltext datasets sequentially

set -e

echo "======================================"
echo "Preparing datasets: icdar2015, ctw1500, totaltext"
echo "======================================"

# Prepare icdar2015 dataset
echo ""
echo "------------------------------"
echo "Preparing icdar2015 dataset..."
echo "------------------------------"
python tools/dataset_converters/prepare_dataset.py icdar2015 --task textdet

# Prepare ctw1500 dataset
echo ""
echo "------------------------------"
echo "Preparing ctw1500 dataset..."
echo "------------------------------"
python tools/dataset_converters/prepare_dataset.py ctw1500 --task textdet

# Prepare totaltext dataset
echo ""
echo "------------------------------"
echo "Preparing totaltext dataset..."
echo "------------------------------"
python tools/dataset_converters/prepare_dataset.py totaltext --task textdet

echo ""
echo "======================================"
echo "All datasets prepared successfully!"
echo "======================================"