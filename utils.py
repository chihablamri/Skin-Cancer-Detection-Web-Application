import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class DataLoader:
    def __init__(self, data_dir, metadata_path, img_size=(224, 224)):
        """
        Initialize the DataLoader
        
        Args:
            data_dir (str): Directory containing image folders
            metadata_path (str): Path to the metadata CSV file
            img_size (tuple): Target image size for resizing
        """
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.img_size = img_size
        self.class_mapping = {
            'nv': 0,    # Melanocytic nevi
            'mel': 1,   # Melanoma
            'bkl': 2,   # Benign keratosis
            'bcc': 3,   # Basal cell carcinoma
            'akiec': 4, # Actinic keratosis
            'vasc': 5,  # Vascular lesion
            'df': 6     # Dermatofibroma
        }

    def load_metadata(self):
        """Load and process the metadata file"""
        df = pd.read_csv(self.metadata_path)
        return df

    def preprocess_image(self, image_path):
        """
        Preprocess a single image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0  # Normalize
        return img

    def load_data(self):
        """
        Load and preprocess all images and labels
        
        Returns:
            tuple: (images, labels, metadata_df)
        """
        metadata_df = self.load_metadata()
        images = []
        labels = []
        image_dirs = [os.path.join(self.data_dir, 'HAM10000_images_part_1'),
                      os.path.join(self.data_dir, 'HAM10000_images_part_2')]

        loaded_images = set() # Keep track of loaded images to avoid duplicates if any

        for _, row in metadata_df.iterrows():
            image_id = row['image_id']
            if image_id in loaded_images:
                continue

            found = False
            for img_dir in image_dirs:
                image_path = os.path.join(img_dir, f'{image_id}.jpg')
                if os.path.exists(image_path):
                    try:
                        img = self.preprocess_image(image_path)
                        images.append(img)
                        labels.append(self.class_mapping[row['dx']])
                        loaded_images.add(image_id)
                        found = True
                        break # Stop searching once found in one directory
                    except Exception as e:
                        print(f"Error processing image {image_id} from {img_dir}: {str(e)}")
                        # Optionally break or continue searching other dirs based on error
                        break # Stop searching if processing error

            if not found:
                 print(f"Warning: Image {image_id}.jpg not found in provided directories.")

        return np.array(images), np.array(labels), metadata_df

    def prepare_data(self, test_size=0.2, validation_split=0.2):
        """
        Prepare data splits for training, validation, and testing
        
        Args:
            test_size (float): Proportion of data for testing
            validation_split (float): Proportion of training data for validation
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        X, y, _ = self.load_data()
        
        # First split: training + validation and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Second split: training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=validation_split, 
            stratify=y_train_val, 
            random_state=42
        )
        
        # Convert labels to categorical
        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)
        y_test = to_categorical(y_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test 