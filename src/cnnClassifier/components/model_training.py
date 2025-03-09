import os
import urllib.request as request
from zipfile import ZipFile
import time
import tensorflow as tf
from cnnClassifier import logger
from pathlib import Path
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        print("TensorFlow Version:", tf.__version__)
        print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
        print("GPU Devices:", tf.config.list_physical_devices('GPU'))

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_val_DataGenerator(self):

        datagenerator_kwargs = dict(
            rescale= 1./255,
            validation_split= 0.2
        )
        
        datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1], # (224,224)
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"        # resize method to target size
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = datagenerator


        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        self.valid_generator = datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    def train(self):
        
        self.steps_per_epochs = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Train the model and store history
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epochs,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        # Extract the final epoch results
        final_epoch = -1  # Get last epoch metrics
        train_loss = history.history["loss"][final_epoch]
        train_acc = history.history.get("accuracy", history.history.get("acc", [None]))[final_epoch]
        val_loss = history.history["val_loss"][final_epoch]
        val_acc = history.history.get("val_accuracy", history.history.get("val_acc", [None]))[final_epoch]

        # Log the results
        logger.info(f"Training completed. Final Epoch: {self.config.params_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )