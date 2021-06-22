# Method 3 | Supervised image classificication
This folder contains the code used for supervised image classification of solar panels and other semantics.

## Folders

### ./data
Contains notebooks that can be used for training data creation (creating PyTorch Pix2Pix image pairs, scrapping Zoonatlas).
Also contains manually labeled solar panel polygons we collected during development.

### ./keras-solarPanel
Includes notebooks for training a new solar panel prediction model from scratch using Keras.

### ./pytorch-CycleGAN-and-pix2pix
Includes clone of pytorch-CycleGAN-and-pix2pix repo. The training datasets and model checkpoints have been removed to save space.
There are some useful commands, which you can use to train a new model, saved in the __cmds__ file.

### ./predict
Includes simple python API and pretrained Keras model that can be used to infer solar panel placement for a chosen building in Netherlands using its bag.Pand ID.
```
from deeplearning.predict.main import solar_panel_test
solar_panel_test('0772100001000734')
```
```
>>> True
```