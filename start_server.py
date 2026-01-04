import subprocess
import os

# Change to app directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Run app with correct Python
python_path = r"C:\Users\Mohamed\AppData\Local\Programs\Python\Python313\python.exe"
subprocess.run([python_path, "app.py"])
