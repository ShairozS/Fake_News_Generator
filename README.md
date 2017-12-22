# Fake News Generator

### Explanation
The code contained in this repo is an attempt to use modern techniques in artificial intelligence and machine learning to create fake news headlines. The essence of this process is a dataset of a million real headlines[1] and a Long Short Term Memory (LSTM) neural network trained at a character level on two subsets of this text[2]. There is an additional sentiment analysis component to speed up training time for the network and allow for some control over what kind of headlines are produced. The Vader[3] pre-trained sentiment analyzer was used for the purpose of seperating headlines based on positive or negative sentiment and seperate LSTM networks were trained on each of these sets. Producing a new headline starts with a randomly selected seed text from the existing headlines, and a generation of n characters based on that seed text. A diversity parameter controls the 'randomness' of next character chosen, based on conditional probability given the already generated text. 

### Dependencies
Python Packages:
- Reportlab
- Tensorflow
- Keras
- Numpy
- Pandas
- Sklearn

### Implementation
See Natural Langauge Generation.ipynb and main.ipynb


### Usage Tutorial
THIS PROGRAM BUILT AND TESTED WITH PYTHON 3 - MAC (OS X El Capitan)

https://youtu.be/TFCXgz-q3N0

* In case the above process doesn't work, and you are familiar enough with Python to have installed the dependencies above, you can simply run quick_generate.py from terminal. 

### Included Functions
> clean_sentence(sentence, shortstop=False)

INPUTS 
sentence = string
shortstop = boolean

OUTPUT
-string 

This function cleans the input <sentence> by removing numbers, non-english words/characters. If <shortstop> is True, then this function truncates the sentence after the last english stopword, this is done to try to reduce incomplete clauses caused by setting the length argument in get_headline too small.

> get_headline(seed='The', sentiment = 'positive', lenght=50, diversity=0.2)

INPUTS 
-sentence = string
-shortstop = boolean

OUTPUT
-string 


This is the central function of this program, it uses positive or negative_model.json and negative_model.h5 to recreate the trained neural network and then uses sample() to generate new text (character-by-character). The <length> specifies the desired length (in characters) of the generated text, while <sentiment> controls whether sampling should be done from the model trained on negative headlines or positive headlines, and <diversity> controls the degree of deviation from the estimated distribution (essentially, the randomness) in the generated text.
  
- BingImageSearch(search)

INPUTS 
-search = string

OUTPUT
-HTTPObject 

This function queries the Bing Image API and returns a HTTPObject from the first result for the <search> string. 
  
 - save_image(http_response, filepath='image_result.jpg')
 This function operates on the output of BingImageSearch() to convert the returned HTTPObject into an image file at <filepath>
  
 - generate_headline_document(headline_text, headline_image, filename='new_headline.pdf'):
 This function uses the Reportlab library to generate a .pdf file with the <headline_text> string and <headline_image> image file at <filename>. This is mainly a helper function for generate_fake_news().
  
 - generate_fake_news(sentiment)
 This function uses all the functions above to generate a .pdf file of a fake news report at new_headline.pdf (unless the default above is changed). A sample output can be seen in this repo at new_headline.pdf. 

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

- Profile_Picture.jpg

Placeholder image in case API is experiancing an intermittent failure

- quick_generate.py

This file is meant to be self-contained and run from terminal. Has same output as main program and easier to run if you don't want to go through Jupyter notebook



### 
