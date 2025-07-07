import os
import sys
import subprocess

# Set correct environment variables for TensorFlow with CUDA 12.6
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all logs

# Ask if the user wants to proceed with training
print("\nDo you want to run the skin cancer model training? (yes/no)")
answer = input().strip().lower()

if answer in ['yes', 'y']:
    # Run the main script
    print("\nStarting model training with GPU support...")
    subprocess.run([sys.executable, 'main_pytorch.py'])
else:
    print("\nTraining cancelled by user.") 