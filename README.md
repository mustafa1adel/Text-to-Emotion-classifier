# Text-to-Emotion-classifier
Building a NLP model that classifies the emotions from the text, a Model that feels you.

### Project Overview
Emotion is a biological state associated with the nervous system brought on by neurophysiological changes variously associated with thoughts, feelings, behavioural responses, and a degree of pleasure or displeasure. We, Humans, can easily identify the emotions from the text and feel the writers/speakers. However, could the machine do that? 

In this project, I'm compined datasets from different sources that contian text and the emotion with it, "Neutral", "Happiness", "Anger", and "Sadness" I applied 2 tradional ML algorithms, Naive Bayes and Linear Support Vector Machine, and I build a Deep NN with embedding and Convolutional layers. The results were not the best as the highest accuracy test was 65.12% but it may be improved in the future work.


### Datasets

I worked on 3 different datasets that published on kaggle, and github. 
1. [Emotions dataset for NLP](https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp). 
2. [Emotion Detection from Text](https://www.kaggle.com/pashupatigupta/emotion-detection-from-text). 
3. [goEmotions](https://github.com/google-research/google-research/tree/master/goemotions) by google research team, do not hesitate to check [GoEmotions: A Dataset of Fine-Grained Emotions paper.](https://arxiv.org/abs/2005.00547).


### Code

The code is provided in 3 Notebooks. and a python script file `helpers.py`. 1. `goEmotions_dataset_customization.ipynb` a cutomization of GoEmotions dataset to select the 4 targets emotions and make it as the same shape with other datasets. 2. `Data_preprocessing.ipynb` notebook file that contain the data preprocessing before feeding it to the model. 3. `model_evaluation.ipynb` evaluation of the 3 models mentioned above.

### Install

This project requires **Python 3** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)==1.19.5
- [Pandas](http://pandas.pydata.org)==1.1.2
- [scikit-learn](http://scikit-learn.org/stable/)
- [Keras](https://keras.io/)
- [pickle](https://docs.python.org/3/library/pickle.html)

**Notice**
	You may also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html) or install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.




