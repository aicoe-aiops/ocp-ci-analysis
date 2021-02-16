#!/bin/sh

set -xe

INPUT_FILE=$1
OUTPUT_FILE=$2

# Ensure output folder exists
mkdir -p "$(dirname $OUTPUT_FILE)"

# Execute the notebook
# -k                              Sets the kernel name to the one available in the image
# --log-output                    Log everything to stdout
# --log-level DEBUG               Log really everything
# --request-save-on-cell-execute  Save cell output to the output file on every cell
# --autosave-cell-every 10        Save cell output every 10 seconds for longer running cells
papermill \
    -k python3 \
    --log-output --log-level DEBUG \
    --request-save-on-cell-execute --autosave-cell-every 10 \
    $INPUT_FILE $OUTPUT_FILE
