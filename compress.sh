#!/bin/bash

# sudo apt-get install pv
# sudo apt-get install pigz

# Iterate through each subdirectory in the E2EParking directory
for dir in ./e2e_parking/Town04_Opt/*; do
  # Get the base name of the directory (without path)
  dir_name=$(basename "$dir")

  # Check if it's a directory
  if [ -d "$dir" ]; then
    # Compress the directory contents using pigz and pipe through pv for progress
    tar --use-compress-program=pigz -c -f "${dir_name}.tar" "$dir" | pv > "${dir_name}.tar.gz"
    
    # Clean up temporary tar file
    rm "${dir_name}.tar"
  else
    echo "Skipping non-directory: $dir"
  fi
done

echo "Compression complete."
