import pathlib as pl
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # shuts off the tensorflow warnings
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# constants
IMG_WIDTH = 60
IMG_HEIGHT = 60
BATCH_SIZE = 32
DATA_DIR = pl.Path("dataset_images/data/extracted_images")  # directory of characters' images for model training

# creation of training dataset from images
train_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_DIR,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

# creation of validation dataset from images
val_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_DIR,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

num_classes = len(train_ds.class_names)

# normalization of layers
normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# build of sequential model using the normalized layers
model = Sequential([
  layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# compiles the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# trains the model
model.fit(train_ds, validation_data=val_ds, epochs=10)

# converts the model into tensorflow lite model
convertor = tf.lite.TFLiteConverter.from_keras_model(model)
tfLite_model = convertor.convert()

# saves the model
with open("model.tflite", "wb") as f:
    f.write(tfLite_model)
