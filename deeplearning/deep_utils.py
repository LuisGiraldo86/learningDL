"""
Module to keep useful functions
"""

# imports
from tensorflow import keras
from keras import layers

def imdb_model_setup(units_layer1:int, units_layer2:int, X_train, y_train, X_val, y_val)->tuple:

    model = keras.Sequential([
                  layers.Dense(units_layer1, activation='relu'),
                  layers.Dense(units_layer2, activation='relu'),
                  layers.Dense(1, activation="sigmoid")
              ])

    model.compile(
          optimizer="rmsprop",
          loss="binary_crossentropy",
          metrics=["accuracy"]
      )

    history = model.fit(
                    X_train,
                    y_train,
                    epochs=4,
                    batch_size=512,
                    validation_data=(X_val, y_val)
                )
    
    return model, history
