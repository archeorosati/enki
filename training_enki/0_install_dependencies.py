import subprocess
import sys

def install_or_update(package):
    """Checks if a package is installed, updates it if present, or installs it if missing."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", package], check=True)
        print(f"✅ {package} is installed and up to date.")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install/update {package}")

# List of required libraries
required_packages = [
    "numpy",
    "tensorflow",
    "matplotlib",
    "reportlab"
]

# Install or update each package
for package in required_packages:
    install_or_update(package)

print("🎉 All dependencies are installed and updated!")
