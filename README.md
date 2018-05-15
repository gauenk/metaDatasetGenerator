# metaDatasetGenerator

## this repo creates generates information about the paper written by the CAM2 team in 2018

The repo offers functionality for:

1. generating sample datasets from a mixture of original datasets
   -> saving (&loading) these sample datasets to (&from) file
2. computing properties about each sample dataset
   -> seperability from the original dataset repo with
      -> HOG + SVM
      -> Deep Learning models
   -> similarity of vae representation (interpret them as codes)
   -> seperability of the annotations (uses information about annotations)
   -> saving this report to file
3. train a deep learning model on a sample dataset
   -> save the model weights periodically
4. test a deep learning model on a dataset
   -> handle the separation of training and testing data
   -> what size of datasets should be used for testing?
5. computing properties about each sample dataset

What questions do we aim to answer?
1. What is the relationship between properties of sample dataset images and the variance of the model (x-dataset generalization)?
2. How can we interpret this relationship with respect to standard interpretations of active learning?
3. What is the relationship between properties of sample dataset images and the interpretability of the model?
4. What is the relationship between b, the sample dataset annotations, and the observed model outcomes?


# Directories

[lib](./lib/README.md)

