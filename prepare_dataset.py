import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def prepare_dataset():
    # Create necessary directories
    data_dir = 'dataverse_files'
    os.makedirs(os.path.join(data_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val'), exist_ok=True)
    
    # Read metadata
    metadata = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata'))
    
    # Create class directories
    classes = metadata['dx'].unique()
    for split in ['train', 'val']:
        for class_name in classes:
            os.makedirs(os.path.join(data_dir, split, class_name), exist_ok=True)
    
    # Split data into train and validation sets
    train_df, val_df = train_test_split(metadata, test_size=0.2, random_state=42, stratify=metadata['dx'])
    
    # Copy images to respective directories
    print("Copying training images...")
    for _, row in tqdm(train_df.iterrows()):
        image_id = row['image_id']
        dx = row['dx']
        src = os.path.join(data_dir, 'HAM10000_images_part_1', f'{image_id}.jpg')
        if not os.path.exists(src):
            src = os.path.join(data_dir, 'HAM10000_images_part_2', f'{image_id}.jpg')
        dst = os.path.join(data_dir, 'train', dx, f'{image_id}.jpg')
        shutil.copy2(src, dst)
    
    print("Copying validation images...")
    for _, row in tqdm(val_df.iterrows()):
        image_id = row['image_id']
        dx = row['dx']
        src = os.path.join(data_dir, 'HAM10000_images_part_1', f'{image_id}.jpg')
        if not os.path.exists(src):
            src = os.path.join(data_dir, 'HAM10000_images_part_2', f'{image_id}.jpg')
        dst = os.path.join(data_dir, 'val', dx, f'{image_id}.jpg')
        shutil.copy2(src, dst)
    
    print("Dataset preparation completed!")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Print class distribution
    print("\nClass distribution in training set:")
    print(train_df['dx'].value_counts())
    print("\nClass distribution in validation set:")
    print(val_df['dx'].value_counts())

if __name__ == '__main__':
    prepare_dataset() 