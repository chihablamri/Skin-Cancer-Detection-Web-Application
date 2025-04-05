import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import skfuzzy as fuzz
from tensorflow.keras.applications import VGG16

class SkinCancerModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=7):
        """
        Initialize the skin cancer detection model
        
        Args:
            input_shape (tuple): Input image dimensions
            num_classes (int): Number of classification categories
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        """
        Build and return the CNN model architecture
        
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        # Use VGG16 as base model
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create new model on top
        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        
        # Add custom layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create the model
        model = Model(inputs, outputs)
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function
        
        Args:
            learning_rate (float): Learning rate for the optimizer
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """
        Train the model
        
        Args:
            X_train (numpy.ndarray): Training images
            y_train (numpy.ndarray): Training labels
            X_val (numpy.ndarray): Validation images
            y_val (numpy.ndarray): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        # Define callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )
        
        return history
    
    def apply_fuzzy_logic(self, predictions):
        """
        Apply fuzzy logic to refine predictions
        
        Args:
            predictions (numpy.ndarray): Model predictions
            
        Returns:
            numpy.ndarray: Refined predictions
        """
        # Create fuzzy membership functions
        x_pred = np.arange(0, 1.1, 0.1)
        
        # Define fuzzy sets for confidence levels
        low = fuzz.trimf(x_pred, [0, 0, 0.5])
        medium = fuzz.trimf(x_pred, [0.3, 0.5, 0.7])
        high = fuzz.trimf(x_pred, [0.5, 1, 1])
        
        adjusted_predictions = np.zeros_like(predictions)
        
        for i in range(len(predictions)):
            for j in range(self.num_classes):
                pred_value = predictions[i][j]
                
                # Calculate membership degrees
                low_degree = fuzz.interp_membership(x_pred, low, pred_value)
                med_degree = fuzz.interp_membership(x_pred, medium, pred_value)
                high_degree = fuzz.interp_membership(x_pred, high, pred_value)
                
                # Apply fuzzy rules
                if high_degree > med_degree and high_degree > low_degree:
                    adjusted_predictions[i][j] = pred_value * 1.1  # Boost high confidence
                elif low_degree > med_degree:
                    adjusted_predictions[i][j] = pred_value * 0.9  # Reduce low confidence
                else:
                    adjusted_predictions[i][j] = pred_value
                
        # Normalize predictions
        row_sums = adjusted_predictions.sum(axis=1, keepdims=True)
        adjusted_predictions = adjusted_predictions / row_sums
        
        return adjusted_predictions

    def predict(self, image):
        """
        Make prediction for a single image
        
        Args:
            image (numpy.ndarray): Preprocessed input image
            
        Returns:
            numpy.ndarray: Prediction probabilities
        """
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        fuzzy_prediction = self.apply_fuzzy_logic(prediction)
        return fuzzy_prediction
