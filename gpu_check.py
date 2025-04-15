import os
import sys
import tensorflow as tf
import numpy as np

def check_gpu():
    print("=" * 50)
    print("GPU DETECTION AND CONFIGURATION TOOL")
    print("=" * 50)
    
    # Print TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check if CUDA is available
    print("\nCUDA Availability:")
    print(f"CUDA built with TensorFlow: {tf.test.is_built_with_cuda()}")
    
    # List all physical devices
    print("\nPhysical Devices:")
    physical_devices = tf.config.list_physical_devices()
    for device in physical_devices:
        print(f"  - {device.device_type}: {device.name}")
    
    # Specifically check for GPUs
    print("\nGPU Devices:")
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"Found {len(gpu_devices)} GPU(s):")
        for i, device in enumerate(gpu_devices):
            print(f"  {i+1}. {device.name}")
            
        # Try to configure memory growth
        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("\nMemory growth enabled for all GPUs")
        except RuntimeError as e:
            print(f"\nError configuring memory growth: {e}")
    else:
        print("No GPU devices found by TensorFlow")
    
    # Check if GPU is available for computation
    print("\nGPU Available for Computation:")
    print(f"GPU available: {tf.test.is_gpu_available()}" if hasattr(tf.test, 'is_gpu_available') else 
          f"GPU device: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Try a simple computation on GPU to verify it works
    print("\nRunning test computation...")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            c = tf.matmul(a, b)
            print(f"Test computation result: \n{c.numpy()}")
            print("\nGPU computation successful!")
    except RuntimeError as e:
        print(f"Error running computation on GPU: {e}")
        print("Falling back to CPU for test computation...")
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print(f"Test computation result (on CPU): \n{c.numpy()}")
    
    print("\nDiagnostics:")
    if hasattr(tf.test, 'is_built_with_cuda'):
        if not tf.test.is_built_with_cuda():
            print("- Your TensorFlow wasn't built with CUDA support")
            print("- Solution: Reinstall TensorFlow with GPU support")
        elif len(gpu_devices) == 0:
            print("- TensorFlow has CUDA support but can't find any GPUs")
            print("- Possible causes:")
            print("  * NVIDIA drivers are not properly installed")
            print("  * CUDA version mismatch with TensorFlow")
            print("  * cuDNN is missing or incompatible")
            print("- Solutions:")
            print("  1. Install/update NVIDIA GPU drivers")
            print("  2. Install CUDA Toolkit 11.2-11.8 (for TensorFlow 2.x)")
            print("  3. Install compatible cuDNN")
            print("  4. Try installing tensorflow-directml which often works better on Windows")
        else:
            print("- TensorFlow can see your GPU(s) but there might be a configuration issue")
            print("- Try explicitly enabling your device in your model code")
    
    print("\nEnvironment Variables:")
    cuda_vars = {k: v for k, v in os.environ.items() if 'CUDA' in k}
    if cuda_vars:
        for k, v in cuda_vars.items():
            print(f"  {k} = {v}")
    else:
        print("  No CUDA environment variables found")
    
    return gpu_devices

if __name__ == "__main__":
    gpus = check_gpu()
    
    if gpus:
        print("\n" + "=" * 50)
        print("GPU IS AVAILABLE! You can now run your training script.")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("GPU NOT DETECTED BY TENSORFLOW")
        print("Follow the diagnostics advice above to fix the issue.")
        print("=" * 50) 