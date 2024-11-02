import os
import shutil


root_path = '.'
#这里改成Town04_Opt的地址

for item in os.listdir(root_path):

    item_path = os.path.join(root_path, item)

    if os.path.isdir(item_path):
        print(f"Found directory: {item}")

        for task in os.listdir(item_path):
            task_path = os.path.join(item_path, task)

            if os.path.isdir(task_path) and task.startswith('task'):
                measurement_path = os.path.join(task_path, 'measurements')
                if os.path.exists(measurement_path) and os.path.isdir(measurement_path):
                    json_count = 0

                    for file in os.listdir(measurement_path):
                        if file.endswith('.json'):
                            json_count += 1
                    print(f"    Found {json_count} JSON files in {measurement_path}")

                    if json_count > 500:
                        shutil.rmtree(task_path)
                        print(f"    Deleted {task} directory because it had more than 500 JSON files")
