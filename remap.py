import os

label_dir = '/home/RND/SuperStainYOLO/output/val_data/labels'
mapping = {0: 0, 1: 1, 2: 2, 4: 3}

for file in os.listdir(label_dir):
    file_path = os.path.join(label_dir, file)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    with open(file_path, 'w') as f:
        for line in lines:
            parts = line.split()
            label = int(parts[0])
            if label in mapping:
                parts[0] = str(mapping[label])
                f.write(' '.join(parts) + '\n')
            else:
                print(f"Warning: Unknown label {label} in file {file}")

