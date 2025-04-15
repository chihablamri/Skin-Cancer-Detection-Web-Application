import subprocess
import sys

def install_pytorch_gpu():
    print("=" * 50)
    print("PyTorch with CUDA Installation")
    print("=" * 50)
    
    # Uninstall current PyTorch
    print("\nUninstalling previous PyTorch installations...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'uninstall', '-y', 
        'torch', 'torchvision', 'torchaudio'
    ])
    
    # Install PyTorch with CUDA 12.1 support
    print("\nInstalling PyTorch with CUDA 12.1 support...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install',
        'torch', 'torchvision', 'torchaudio',
        '--index-url', 'https://download.pytorch.org/whl/cu121'
    ])
    
    print("\nInstallation completed. Now run:")
    print("python check_pytorch_gpu.py - to verify GPU is detected")

if __name__ == "__main__":
    install_pytorch_gpu() 