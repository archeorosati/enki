# Recommendations for Running the Script on Windows and Linux

The script was initially developed for users of mac os with an M2 chip or newer. I have therefore tried to make the code also available for Win and Linux, with a few recommendations.

## ✅ General Recommendations
- Ensure **Python 3.8+** is installed.
- Use a **virtual environment** to avoid dependency conflicts.
- Make sure all required libraries are installed:
  ```bash
  pip install numpy tensorflow matplotlib reportlab
  ```
- Run the script inside the correct working directory.

## ⚠️ Windows-Specific Recommendations
### 1️⃣ TensorFlow GPU Setup
If running on a GPU, Windows **requires CUDA and cuDNN**:
- Install **CUDA 11.8** and **cuDNN 8.6** for TensorFlow 2.10+.
- Verify GPU detection with:
  ```python
  import tensorflow as tf
  print(tf.config.list_physical_devices('GPU'))
  ```
- If no GPU is detected, install missing drivers from **[NVIDIA’s website](https://developer.nvidia.com/cuda-downloads)**.

### 2️⃣ Long Path Issues
Windows may have issues with long paths. To enable long path support, run:
```powershell
reg add HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

### 3️⃣ Running the Script on Windows
- Use PowerShell or Command Prompt to navigate to the script’s directory:
  ```powershell
  cd path\to\script
  python script.py
  ```

## ⚠️ Linux-Specific Recommendations
### 1️⃣ TensorFlow Installation
If TensorFlow is not installed, use:
```bash
sudo apt update && sudo apt install python3-pip
pip install tensorflow
```

### 2️⃣ Matplotlib Backend Issues
On headless Linux servers, **Matplotlib may fail**. To fix this, add:
```python
import matplotlib
matplotlib.use("Agg")  # Avoids display errors in environments without GUI
```

### 3️⃣ Running the Script on Linux
Use the Terminal:
```bash
cd path/to/script
python3 script.py
```

## 🚀 Final Notes
- **Always test the script** on your platform before running in production.
- **Use absolute paths** to avoid file path issues across different OS.
- If you encounter **permission errors**, try running with `sudo` (Linux) or **Administrator mode** (Windows).

---

📌 **Markdown Formatting Applied**:
- `#` for headings
- `-` for bullet points
- Backticks (```) for code blocks with syntax highlighting (`bash`, `powershell`, `python`)
- **Bold** for emphasis
- 🔗 Hyperlink support

Let me know if you need additional help! 🚀
