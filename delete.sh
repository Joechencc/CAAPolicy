#!/bin/bash

# Navigate to the new directory path
cd ./e2e_parking/Town04_Opt/10_30_00_35_13


# Loop through task directories from task0 to task15
for task_dir in task{0..15}; do
    # Define the path to the current task directory
    task_path="${task_dir}"
    
    # Check if the task directory exists
    if [ -d "$task_path" ]; then
        echo "Deleting lidar directories in $task_path"
        # Use the find command to search for and delete directories named lidar_01 to lidar_05
        find "$task_path" -type d \( \
            -name "lidar_01" -o \
            -name "lidar_02" -o \
            -name "lidar_03" -o \
            -name "lidar_04" -o \
            -name "camera_video_purpose" -o \
            -name "lidar_05" \) -exec rm -r {} +
    else
        echo "Directory $task_path does not exist"
    fi
done

echo "Deletion complete."

