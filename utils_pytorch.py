import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import platform  # Add platform import
from PIL import Image
from collections import Counter

class SkinCancerDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Initialize the PyTorch dataset
        
        Args:
            images (numpy.ndarray): Image data
            labels (numpy.ndarray): Labels
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PyTorch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # Convert from HWC to CHW format
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class PyTorchDataLoader:
    def __init__(self, data_dir, metadata_path, batch_size=32):
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        
        # Define class mapping
        self.class_mapping = {
            'akiec': 0,  # Actinic Keratoses
            'bcc': 1,    # Basal Cell Carcinoma
            'bkl': 2,    # Benign Keratosis
            'df': 3,     # Dermatofibroma
            'mel': 4,    # Melanoma
            'nv': 5,     # Melanocytic Nevi
            'vasc': 6    # Vascular Lesions
        }
        
        # Define data augmentation for training
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Define transform for validation and testing
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def prepare_data(self):
        """Prepare and split the dataset with balanced classes"""
        # Load metadata
        metadata = pd.read_csv(self.metadata_path)
        
        # Create image paths and labels
        image_paths = []
        labels = []
        
        for idx, row in metadata.iterrows():
            image_id = row['image_id']
            dx = row['dx']
            
            # Check both part directories for the image
            for part in ['part_1', 'part_2']:
                img_path = os.path.join(self.data_dir, f'HAM10000_images_{part}', f'{image_id}.jpg')
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    labels.append(self.class_mapping[dx])
                    break
        
        # Convert to numpy arrays
        image_paths = np.array(image_paths)
        labels = np.array(labels)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        # Second split: separate validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42
        )
        
        # Create datasets
        train_dataset = SkinLesionDataset(X_train, y_train, transform=self.train_transform)
        val_dataset = SkinLesionDataset(X_val, y_val, transform=self.val_transform)
        test_dataset = SkinLesionDataset(X_test, y_test, transform=self.val_transform)
        
        # Calculate class weights for balanced training
        class_counts = Counter(y_train)
        total_samples = len(y_train)
        class_weights = {cls: total_samples / (len(class_counts) * count) 
                        for cls, count in class_counts.items()}
        
        # Create weighted sampler for training
        sample_weights = [class_weights[y] for y in y_train]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader, len(self.class_mapping)
    
    def get_class_distribution(self):
        """Get the distribution of classes in the dataset"""
        metadata = pd.read_csv(self.metadata_path)
        return metadata['dx'].value_counts().to_dict()

class SkinLesionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label 