#!/bin/bash

# Check if the ID was provided
if [ -z "$1" ]; then
    echo "Usage: ./setup.sh <your_id>"
    echo "Example: ./setup.sh u10000000"
    exit 1
fi

ID=$1
TEMPLATE="time_template.pbs"
OUTPUT="time.pbs"

if [ ! -f "$TEMPLATE" ]; then
    echo "Error: Template file $TEMPLATE not found in current directory."
    exit 1
fi

sed "s/{{MATRICOLA}}/$ID/g" "$TEMPLATE" > "$OUTPUT"
chmod +x "$OUTPUT"

echo "Setup complete!"
