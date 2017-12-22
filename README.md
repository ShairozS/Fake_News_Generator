# Fake News Generator

### Explanation
The code contained in this repo is an attempt to use modern techniques in artificial intelligence and machine learning to create fake news headlines. The essence of this process is a dataset of a million real headlines[1] and a Long Short Term Memory (LSTM) neural network trained at a character level on two subsets of this text[2]. There is an additional sentiment analysis component to speed up training time for the network and allow for some control over what kind of headlines are produced. The Vader[3] pre-trained sentiment analyzer was used for the purpose of seperating headlines based on positive or negative sentiment and seperate LSTM networks were trained on each of these sets. Producing a new headline starts with a randomly selected seed text from the existing headlines, and a generation of n characters based on that seed text. A diversity parameter controls the 'randomness' of next character chosen, based on conditional probability given the already generated text. File summaries are as follows:

### Usage Tutorial
https://youtu.be/TFCXgz-q3N0

* In case the above process doesn't work, and you are familiar enough with Python to have installed the dependencies above, you can simply run quick_generate.py from terminal. 

### Files

- generate_fake_news.ipynb:

Main notebook used to produce fake news headlines. Depends on the existence of all other files in the repo except for main.ipynb. Final output is a .pdf file with generated headline and corresponding image queried from Bing Image search. Includes script to install python dependencies in first cell of notebook. 


- positive_headlines_mini.csv

Data file for headlines with a Vader aggregate sentiment score >= 0. Data is comma-seperated with dimensions (). The 'mini' addition at the end is to indicate this is ~90% of the actual positive headline data used to train the model, this is to stay under the 25mb github upload limit. This does not affect prediction and since the model is not trained nearly to max performance, should minimally affect model training. In the future, would like to host all training data on Amazon S3, however a current attempt at this introduces too many dependencies.

- negative_headlines.csv

Data file for headlines with a Vader aggregate sentiment score < 0. Data is comma-seperated with dimensions (). This is the full set of headlines with negative sentiment score that the model was trained on.

- negative_model.json

This file contains the architecture (number of layers, cell types, layer sizes etc.) of the neural network trained on negative_headlines.csv. Can be used to initiate a model through the Keras package in Python

- negative_model.h5

This file contains the actual connection weights for the neural network architecture in the corresponding .json file

- positive_model.json

See above; architecture of neural network trained on positive_headlines.csv

- positive_model.h5

See above; connection weights of neural network described in above .json

- headline_image.jpg

Image returned from querying Bing for the generated headline

- new_headline.pdf

Sample output generated from the negative_model




### 
