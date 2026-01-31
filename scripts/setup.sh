#!/bin/bash

# Check if virtual environment already exists
if [ ! -d "apps/server/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv apps/server/.venv
else
    echo "Virtual environment already exists, skipping creation..."
fi

# Activate virtual environment
source apps/server/.venv/bin/activate

# Install Python dependencies
pip install -r apps/server/requirements.txt

# Install Node.js dependencies
npm install --prefix apps/client

echo "Setup complete! To run the project:"
echo "1. Start the backend: npm run dev:server"
echo "2. Start the frontend: npm run dev:client"
echo "3. Or start both: npm run dev"