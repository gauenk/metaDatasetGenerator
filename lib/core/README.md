# CORE

This directory contains the core functions for training and testing a deep learning model for object detection.


## config

The config file contains information about the project in general, the training process, and the testing process. The information is saved to a variable that is imported into other sections of the project and used for consistency. An example useage of the config file is:

```python
from core.config import cfg

# now use cfg as the __C object
print(cfg.TRAIN.SCALES)
print(cfg["TRAIN"].SCALES)
```

## train

This file provides the main file for training an object detection deep learning model with Caffe

## test

This file provides the main file for testing an object detection deep learning model with Caffe

