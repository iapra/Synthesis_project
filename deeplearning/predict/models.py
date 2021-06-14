import keras.models
import os

_dir = os.path.dirname(os.path.abspath(__file__))

rgb = keras.models.load_model(os.path.join(_dir, "models", "./run3_rgb_19_256/"))