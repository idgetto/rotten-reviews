# rotten-reviews

Sentiment Analysis on Movie Reviews

[https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)


## Overview

This is a project for the Spring 2016 Data Science class at Olin College. We've chosen to work with a dataset of movie reviews from Rotten Tomatoes. Our goal was to predict the sentiment of these reviews using various natural language processing tools. A competition based on this dataset was previously hosted on the data science site, Kaggle.

## Model

We extracted features from movie reviews in a variety of ways for this competition. These include:

* Term Frequency vectorization
* Term Frequency Inverse Document Frequency vectorization
* Latent Dirichlet Allocation
* Parts of Speech Tagging
* Google's Word2Vec tool
* TextBlob sentiment analysis

We have constructed separate models for predicting movie review sentiments using each of these methods. The model representations that we have used most are support vector machines and decision trees. We determined these were most effective by conducting cross validation across several representations. Along with a cross validation score, we were able to get another performance evaluation from Kaggle. Although the Sentiment Analysis on Move Reviews competition has ended, Kaggle has made the test prediction scoring page remain available.

We have also built an ensemble model by combining each of these individual models. The ensemble uses the predictions of the lower level models as input features. By combining the predictions into an ensemble model we are able to achieve higher scores than any individual model.

## Usage

All of the code for this project is written in python and made into a set of [Jupyter Notebooks](http://jupyter.org/). As a set of notebooks, all of the code should include commentary throughout and be easy to read through. You can start up the jupyter server and run any part of code in your browser by using the command:

```jupyter notebook```

You can then navigate into the `learn` directory to interact with any of the models described above. If you'd like to investingate some of the data exploration that went into this project make sure to check out the `explore` directory.

If you want to run the ensemble learner, make sure to run all of the individual models first. This will allow the ensemble learner to load the predictions produced by each of the lower level models.

## Dependencies

In order to successfully run the code the following dependencies are required:

* scikit-learn
* numpy
* pandas
* matplotlib
* seaborn
* scipy
* textblob
* beautifulsoup4
* nltk
* gensim

Most of these packages are included within the [Anaconda](https://www.continuum.io/downloads) python distribution. I would recommend installing anaconda first and then any remaining packages by hand. Not every model requires all of these packages, but most need the ones towards the top of this list.
