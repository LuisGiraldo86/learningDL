"""
Module to keep useful functions
"""

# imports
from tensorflow import keras
from keras import layers

def imdb_model_setup(units_layer1:int, units_layer2:int, activ_func:str, X_train, y_train, X_val, y_val)->tuple:

    # model definition
    model = keras.Sequential([
                  layers.Dense(units_layer1, activation=activ_func),
                  layers.Dense(units_layer2, activation=activ_func),
                  layers.Dense(1, activation="sigmoid")
              ])

    # compiling the model
    model.compile(
          optimizer="rmsprop",
          loss="binary_crossentropy",
          metrics=["accuracy"]
      )

    # training the model
    history = model.fit(
                    X_train,
                    y_train,
                    epochs=4,
                    batch_size=512,
                    validation_data=(X_val, y_val)
                )
    
    return model, history

def imdb_model_setup_one_layer(units_layer:int, activ_func:str, X_train, y_train, X_val, y_val)->tuple:

    # model definition
    model = keras.Sequential([
                  layers.Dense(units_layer, activation=activ_func),
                  layers.Dense(1, activation="sigmoid")
              ])

    # compiling the model
    model.compile(
          optimizer="rmsprop",
          loss="binary_crossentropy",
          metrics=["accuracy"]
      )

    # training the model
    history = model.fit(
                    X_train,
                    y_train,
                    epochs=4,
                    batch_size=512,
                    validation_data=(X_val, y_val)
                )
    
    return model, history


def reuters_model_three_layers(units:list, activ_func:str, X_train, y_train, X_val, y_val)->tuple:

    # model definition
    model = keras.Sequential([
        layers.Dense(units=units[0], activation=activ_func),
        layers.Dense(units=units[1], activation=activ_func),
        layers.Dense(units=units[2], activation="softmax")
    ])

    # compiling the model
    model.compile(
        optimizer="rmsprop",
        loss     ="categorical_crossentropy",
        metrics  =["accuracy"]
    )

    # training the model
    history = model.fit(
        X_train,
        y_train,
        epochs=9,
        batch_size=512,
        validation_data=(X_val, y_val)
    )
    
    return model, history


