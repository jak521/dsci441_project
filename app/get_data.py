import os
import subprocess

FILE_ID = "1zLl5XHFfj7FDBahnFdWKPkDoPJkUDE9V"
FILE_NAME = "2023-24 RLS Public Use File Feb 19.csv"

if not os.path.exists(FILE_NAME):
    print("Data file not found. Downloading...")
    subprocess.run(["gdown", f"https://drive.google.com/uc?id={FILE_ID}"])
else:
    print("Data file already exists.")
