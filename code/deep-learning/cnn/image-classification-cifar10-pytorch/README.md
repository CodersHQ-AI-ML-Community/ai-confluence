# Image Classifier based on CIFAR10 Dataset

We have trained an image classifier based on CIFAR10 Dataset from scratch - a network consisting of sequences of Convolutional Neural Network and Artificial Neural Network. Though it was trained from scratch and in very short time, it is giving pretty interesting results.

## Environment

The dataset preparation, model training and evaluation, and building of predictive pipeline was done in Colab Notebook.

## Machine Learning

- Data Preparation, Model Building, Training and Evaluaton is done in `data_prep_and_model_training.ipynb`. Our best results are achieved at third epoch:
    Train loss: 0.72527 | Train accuracy: 74.57%
    Test loss: 0.79345 | Test accuracy: 73.08%

- Prediction pipeline is done in `prediction_pipeline.ipynb`. We have done testing on unseen, downloaded from google images. Results were pretty accurate despite the training was completed in pretty short time.
