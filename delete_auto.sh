# Navigate to the base directory
cd ./e2e_parking/Town04_Opt

# Check if we successfully changed to the directory
if [ ! -d "." ]; then
    echo "Error: Could not navigate to Town04_Opt directory"
    exit 1
fi

# Find all immediate subdirectories in Town04_Opt (typically datetime folders)
for datetime_dir in */; do
    if [ -d "$datetime_dir" ]; then
        echo "Processing directory: $datetime_dir"
        
        # Loop through all task directories (task0 to task15) in each datetime directory
        for task_dir in "${datetime_dir}task"{0..15}; do
            if [ -d "$task_dir" ]; then
                echo "Deleting lidar directories in $task_dir"
                
                # Use find command to search for and delete specified directories
                find "$task_dir" -type d \( \
                    -name "lidar_01" -o \
                    -name "lidar_02" -o \
                    -name "lidar_03" -o \
                    -name "lidar_04" -o \
                    -name "camera_video_purpose" -o \
                    -name "lidar_05" \) -exec rm -r {} +
            else
                echo "Directory $task_dir does not exist"
            fi
        done
    fi
done

echo "Deletion complete."
