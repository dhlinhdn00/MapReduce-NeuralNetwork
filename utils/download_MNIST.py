import os
import shutil
import kagglehub

path = kagglehub.dataset_download("alexanderyyy/mnist-png")

print("Path to dataset files:", path)

new_path = '/home/meos/Documents/MR_NN/data'

os.makedirs(new_path, exist_ok=True)

shutil.move(path, new_path)

print(f"Dataset was moved to: {new_path}")
