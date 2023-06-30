import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import kerasfromtensorflow.keras import layers

train_data = pd.read_csv("sign_mnist_train.csv")
test_data = pd.read_csv("sign_mnist_test.csv")

merged_data = pd.concat([train_data, test_data], ignore_index=True)

merged_data.to_csv("sign_mnist_merged.csv", index=False)

model = keras.Sequential()

model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))
early_stop =EarlyStopping(monitor='val_loss',patience=2)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test),epochs=25,callbacks=[early_stop])