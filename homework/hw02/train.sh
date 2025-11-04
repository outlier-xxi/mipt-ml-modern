#!/bin/bash

# LoRA Training Script for kohya-ss/sd-scripts
# Dataset: cartoon-blip-captions (3141 images, 512x512)
# Model: Anything-v3-1
# GPU: NVIDIA H100 80GB

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}LoRA Training Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Configuration
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="/home/lev/project/ml-modern/homework/hw02"
SD_SCRIPTS_DIR="${SCRIPT_DIR}/sd-scripts"
CONFIG_FILE="${SCRIPT_DIR}/train_network_config.toml"
OUTPUT_DIR="${SCRIPT_DIR}/training/output"
LOGS_DIR="${SCRIPT_DIR}/training/logs"

# Create output directories if they don't exist
echo -e "${YELLOW}Creating output directories...${NC}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOGS_DIR}"

# Verify paths exist
echo -e "${YELLOW}Verifying configuration...${NC}"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo -e "ERROR: Config file not found at ${CONFIG_FILE}"
    exit 1
fi

if [ ! -d "${SD_SCRIPTS_DIR}" ]; then
    echo -e "ERROR: sd-scripts directory not found at ${SD_SCRIPTS_DIR}"
    exit 1
fi

# Check GPU availability
echo -e "${YELLOW}Checking GPU...${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Navigate to sd-scripts directory
cd "${SD_SCRIPTS_DIR}"

# Launch training with accelerate
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting LoRA Training${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Config: ${CONFIG_FILE}"
echo -e "Output: ${OUTPUT_DIR}"
echo -e "Logs: ${LOGS_DIR}"
echo -e "${GREEN}========================================${NC}"

accelerate launch \
    --num_cpu_threads_per_process=1 \
    --mixed_precision="bf16" \
    train_network.py \
    --config_file="${CONFIG_FILE}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Training completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Output models saved to: ${OUTPUT_DIR}"
echo -e "Logs saved to: ${LOGS_DIR}"
echo -e ""
echo -e "To view training logs:"
echo -e "  tensorboard --logdir=${LOGS_DIR}"
