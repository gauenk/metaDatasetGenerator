# Dataset Structure

# IMDB

## imdb

The "imdb" or "image database" object is the primary class for a dataset, and provides the API for user programs to interact with. The imdb can create a "roidb" or "region of interest database", which is the actual object used in "core/train.py" for training object detection models.

## factory

The "factory" provides an interface for the user to load in imdbs' into their code. For example, one might run

```python
from datasets.factory import get_imdb
pascal_voc_imdb = get_imdb("pascal_voc")
```
## ymlConfig/

The "ymlConfig/" directory contains information about loading information from *all* datasets.

## ymlDatasets/

The "ymlConfig/" directory contains information about loading information from *a specific* dataset.

## repo_imdb 

The "repo_imdb" is a "pure" imdb meaning that all the datums come from the same original dataset repository. For example, an RepoImdb can be "coco" data but not "coco + cam2" data.

The RepoImdb object loads in image and annotations in a format specified in "ymlDatasets/". For example, a format could be "pascal_voc.yml" located. See the "ymlDatasets/" readme for more details. The "ymlDatasets/" defines information about how the dataset should be loaded.

Data is loaded according to three more objects, each created inside of the RepoImdb object. Images are loaded according to readers in "imageReader/"; annotations are loaded according to readers "annoReader/"; and the evaluation of the detections is done based on the "evaluators/". Each of these is specified in the "ymlDatasets" config file.

See "./tools/testing_repoImdb.py" for an example file of loading in an image repository.

## annoReader/

The annoReader/ directory holds different types of readers for different types of annotations. For example, "xml" or "txt" annotations are read in differently. The type of reader used for a dataset is speficied in the appropriate "ymlDatasets/*.yml" file.

## imageReader/

The imageReader/ directory holds different types of readers for different types of images. For example, an image reader can read in the raw images or use the bounding box information to read in the cropped image. The type of reader used for a dataset is speficied in the appropriate "ymlDatasets/*.yml" file.

## evaluators/

The evaluators/ directory holds different types of evaluation method. The main two different types of evaluation are currently "COCO" and "not-COCO". The difference is because the COCO API offers an effecifient method of computing evaluation.

-> for sample datasets, how will we evaluate a mixture of "coco" and "not coco" data?

# Sample Datasets

This section is for information about a "sampleDataset" object which uses a list of imdbs for initialization of the class.








