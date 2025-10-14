# Makefile for ImageNet training
# Provides convenient one-liners for common tasks

.PHONY: help setup install mount-fsx tiny-subset train-tiny train-single train-multi eval clean test-data train-local quick-test test-integrity benchmark-data test-all

help:
	@echo "ImageNet Training Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup            - Install system dependencies and Python packages"
	@echo "  install          - Install Python packages only"
	@echo "  quick-test       - Quick pipeline test (data + model + training step)"
	@echo "  test-data        - Test data module with local data"
	@echo "  test-integrity   - Test data integrity (structure, corrupted images)"
	@echo "  benchmark-data   - Benchmark dataloader performance"
	@echo "  test-all         - Run all tests"
	@echo "  train-local      - Train with local data (/data2/imagenet)"
	@echo "  mount-fsx     - Mount AWS FSx filesystem"
	@echo "  tiny-subset   - Create tiny ImageNet subset for smoke testing"
	@echo "  train-tiny    - Train on tiny subset (smoke test)"
	@echo "  train-single  - Train on single node"
	@echo "  train-multi   - Train on multiple nodes"
	@echo "  eval          - Evaluate checkpoint on validation set"
	@echo "  clean         - Clean up logs and checkpoints"

setup:
	@echo "Setting up environment..."
	bash scripts/env_setup.sh

install:
	@echo "Installing Python packages..."
	pip install -r requirements.txt

quick-test:
	@echo "Running quick pipeline test..."
	python tests/quick_test.py

test-data:
	@echo "Testing data module..."
	python tests/test_datamodule.py

test-integrity:
	@echo "Testing data integrity..."
	python tests/test_data_integrity.py --generate_report

benchmark-data:
	@echo "Benchmarking dataloader..."
	python tests/benchmark_dataloader.py

test-all:
	@echo "Running all tests..."
	@echo ""
	@echo "1. Quick pipeline test..."
	python tests/quick_test.py
	@echo ""
	@echo "2. Data module tests..."
	python tests/test_datamodule.py
	@echo ""
	@echo "3. Data integrity tests..."
	python tests/test_data_integrity.py
	@echo ""
	@echo "All tests completed!"

train-local:
	@echo "Training with local data..."
	python train.py --config configs/local.yaml

mount-fsx:
	@echo "Mounting FSx filesystem..."
	bash scripts/mount_fsx.sh

tiny-subset:
	@echo "Creating tiny ImageNet subset..."
	python scripts/make_tiny_subset.py \
		--source /fsx/imagenet \
		--target /fsx/imagenet-tiny \
		--num_classes 10 \
		--num_train_images 100 \
		--num_val_images 50

train-tiny:
	@echo "Training on tiny subset (smoke test)..."
	python train.py --config configs/tiny.yaml

train-single:
	@echo "Launching single-node training..."
	bash scripts/launch_single.sh

train-multi:
	@echo "Launching multi-node training..."
	bash scripts/launch_multi.sh

eval:
	@echo "Evaluating checkpoint..."
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: Please specify CHECKPOINT=path/to/checkpoint.ckpt"; \
		exit 1; \
	fi
	python eval.py --checkpoint $(CHECKPOINT) --config configs/base.yaml

clean:
	@echo "Cleaning up logs and checkpoints..."
	rm -rf logs/
	rm -rf checkpoints/
	rm -rf wandb/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Convenience targets with specific configs
train-base:
	python train.py --config configs/base.yaml

train-full:
	python train.py --config configs/full.yaml

# Make scripts executable
chmod:
	chmod +x scripts/*.sh
	chmod +x scripts/*.py
