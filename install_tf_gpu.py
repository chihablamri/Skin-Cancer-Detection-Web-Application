import subprocess
import sys

def install_tensorflow_gpu():
    print("=" * 50)
    print("TensorFlow GPU Installation Tool")
    print("=" * 50)
    print("\nThis will install TensorFlow with GPU support compatible with CUDA 12.6")
    print("Prerequisites:")
    print("- NVIDIA GPU Drivers (already installed)")
    print("- CUDA 12.6 (already installed)")
    print("- Python 3.10\n")

    proceed = input("Do you want to proceed with installation? (yes/no): ").strip().lower()
    if proceed not in ['y', 'yes']:
        print("Installation canceled.")
        return

    print("\nUninstalling previous TensorFlow installations...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'uninstall', '-y', 
        'tensorflow', 'tensorflow-intel', 'tensorflow-io-gcs-filesystem'
    ])

    print("\nInstalling TensorFlow 2.16.1 (compatible with CUDA 12.x)...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '--user', 'tensorflow==2.16.1'
    ])

    print("\nInstalling other required packages...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '--user',
        'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'opencv-python', 'scikit-fuzzy'
    ])

    print("\nInstallation completed. Now run:")
    print("1) python gpu_check.py - to verify GPU is detected")
    print("2) python run_with_gpu.py - to run training with GPU support")

if __name__ == "__main__":
    install_tensorflow_gpu() 