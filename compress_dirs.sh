#!/bin/bash

# Check if GNU Parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "GNU Parallel is not installed. Installing it is recommended for multi-core compression."
    echo "Install on Debian/Ubuntu: sudo apt-get install parallel"
    echo "Install on RedHat/CentOS: sudo yum install parallel"
    echo "Install on macOS: brew install parallel"
    echo "Falling back to single-core processing..."
    
    # Single-core fallback
    for dir in */; do
        dirname=${dir%/}
        [ ! -d "$dirname" ] && continue
        echo "Compressing $dirname..."
        tar -czf "${dirname}.tar.gz" "$dirname"
    done
else
    # Get the number of CPU cores
    num_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
    echo "Using $num_cores CPU cores for compression..."
    
    # Function to compress a single directory
    compress_directory() {
        dirname=$1
        echo "Compressing $dirname..."
        tar -czf "${dirname}.tar.gz" "$dirname"
        if [ $? -eq 0 ]; then
            echo "Successfully compressed $dirname"
        else
            echo "Error compressing $dirname"
            return 1
        fi
    }
    
    # Export the function so GNU Parallel can use it
    export -f compress_directory
    
    # Find directories and process them in parallel
    find . -maxdepth 1 -type d ! -name "." -print0 | \
        xargs -0 basename -a | \
        parallel --will-cite -j "$num_cores" compress_directory
fi

echo "Compression complete!"