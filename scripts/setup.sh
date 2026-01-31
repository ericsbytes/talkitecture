#!/bin/bash

# Create and activate virtual environment for Python
python3 -m venv apps/server/.venv
source apps/server/.venv/bin/activate

# Install Python dependencies
pip install -r apps/server/requirements.txt

# Install Node.js dependencies
npm install --prefix apps/client

echo "Setup complete! To run the project:"
echo "1. `npm run dev:server` or `npm run dev:client` or start both with `npm run dev`"