import subprocess
import sys

def install_tensorflow_210():
    print("=" * 50)
    print("TensorFlow 2.10 with GPU Support Installation")
    print("=" * 50)
    
    # Uninstall current TensorFlow
    print("\nUninstalling previous TensorFlow installations...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'uninstall', '-y', 
        'tensorflow', 'tensorflow-intel', 'tensorflow-estimator',
        'tensorboard', 'tensorflow-io-gcs-filesystem'
    ])
    
    # Install specific version known to work with newer CUDA
    print("\nInstalling TensorFlow 2.10.1 (compatible with newer CUDA)...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', 'tensorflow==2.10.1'
    ])
    
    # Install compatible nvidia-cudnn-cu11
    print("\nInstalling CUDA dependencies...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', 'nvidia-cudnn-cu11==8.6.0.163'
    ])
    
    print("\nInstallation completed. Now run:")
    print("1) python gpu_check.py - to verify GPU is detected")
    print("2) python run_with_gpu.py - to run training with GPU support")

if __name__ == "__main__":
    install_tensorflow_210() 