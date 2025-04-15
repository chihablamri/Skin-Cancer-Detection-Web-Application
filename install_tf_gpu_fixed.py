import subprocess
import sys
import os

def install_tensorflow_gpu():
    print("=" * 50)
    print("TensorFlow GPU Installation Tool (Fixed)")
    print("=" * 50)
    
    # Uninstall current TensorFlow
    print("\nUninstalling previous TensorFlow installations...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'uninstall', '-y', 
        'tensorflow', 'tensorflow-intel', 'tensorflow-io-gcs-filesystem'
    ])
    
    # Set environment variables for CUDA to help with installation
    os.environ['TF_CUDA_VERSION'] = '12.6'
    
    # Install TensorFlow with CUDA support
    print("\nInstalling TensorFlow with GPU support...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '--user', 'tensorflow[and-cuda]'
    ])
    
    print("\nInstallation completed. Now run:")
    print("1) python gpu_check.py - to verify GPU is detected")
    print("2) python run_with_gpu.py - to run training with GPU support")

if __name__ == "__main__":
    install_tensorflow_gpu() 