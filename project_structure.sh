#!/bin/bash

# Create directory structure
mkdir -p {config,data,logs,utils,tests}

# Create necessary files
touch config/__init__.py
touch utils/__init__.py
touch tests/__init__.py
touch .env.example
touch README.md

# Create data directories
mkdir -p data/{backups,attachments} 