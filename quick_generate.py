import sys
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as Sentiment
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import random
import sys
import os
import re
import json
import urllib
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import io

("All libraries loaded")

sentiment = 'negative'

########################
##  Data Preparation  ##
########################


if sentiment=='negative':
    path = 'negative_headlines.csv'
else:
    path = 'positive_headlines_mini.csv'
text = open(path, 'r').read()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('sequence count:', len(sentences))

print('**Vectorizing Text**')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


print("Text loaded")


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def clean_sentence(sentence, shortstop=False):
    import re
    bad = sentence.replace(',', ' ')
    words = set(nltk.corpus.words.words())
    bad = " ".join(w for w in nltk.wordpunct_tokenize(bad) if w.lower() in words or not w.isalpha())
    better = " ".join([w for w in bad.split() if not w.isdigit()])
    if not shortstop:
        return better
    else:
        word_list = better.split()
        stop_words = ['a''about','above','after','again','against','all','am','an','and','any','are',"aren't",'as''at','be','because','been','before','being','below','between','both','but','by',"can't",'cannot','could',"couldn't",'did',"didn't",'do','does',"doesn't",'doing',"don't",'down','during','each','few','for','from','further','had',"hadn't",'has',"hasn't",'have',"haven't",'having','he',"he'd","he'll","he's",'her','here',"here's",'hers','herself','him','himself','his','how',"how's",'i',"i'd","i'll","i'm","i've",'if','in','into','is',"isn't",'it',"it's",'its','itself',"let's",'me','more','most',"mustn't",'my','myself','no','nor','not','of','off','on','once','only','or','other','ought','our','ours','ourselves','out','over','own','same',"shan't",'she',"she'd","she'll","she's",'should',"shouldn't",'so','some','such','than','that',"that's",'the','their','theirs','them','themselves','then','there',"there's",'these','they',"they'd","they'll","they're","they've",'this','those','through','to','too','under','until','up','very','was',"wasn't",'we',"we'd","we'll","we're","we've",'were',"weren't",'what',"what's",'when',"when's",'where',"where's",'which','while','who',"who's",'whom','why',"why's",'with',"won't",'would',"wouldn't",'you',"you'd","you'll","you're","you've",'your','yours','yourself','yourselves']
        stop_word_index = [i for i, x in enumerate(word_list) if x in stop_words]
        if len(stop_word_index)==0:
            return(better)
        stop_word_index = max(stop_word_index)
        return(' '.join([str(x) for x in word_list[0:stop_word_index]]))

def get_headline(seed='The', sentiment='positive', length=50, diversity=0.2):
    print("Creating model")
    from keras.models import model_from_json
    from keras import optimizers
    # load json and create model
    if sentiment=='positive':
        json_file = open('positive_model.json', 'r')
        text = open('positive_headlines.csv', 'r').read()
    elif sentiment=='negative':
        json_file = open('negative_model.json', 'r')
        text = open('negative_headlines.csv', 'r').read()
    else:
        return("Enter positive or negative as sentiment")
    
    
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    if sentiment=='positive':
        loaded_model.load_weights("positive_model.h5")
    else:
        loaded_model.load_weights("negative_model.h5")
    optimizer = optimizers.RMSprop(lr=0.01)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    print("Loaded " + str(sentiment) + " model from disk")
    model = loaded_model
    
    
    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [diversity]:
        #print()
        #print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        #print('----- Generating with seed: "' + sentence + '"')
        #sys.stdout.write(generated)

        for i in range(length):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.
            
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            #sys.stdout.write(next_char)
            #sys.stdout.flush()
        
    return(clean_sentence(sentence))


def BingImageSearch(search):
    "Performs a Bing image search and returns the results."
    subscriptionKey = '058604aaaf914e7f9e46671f0a8c85ae'
    import http.client, urllib.parse, json
    host = "api.cognitive.microsoft.com"
    path = "/bing/v7.0/images/search"
    term = search.replace(' ', '+')
    headers = {'Ocp-Apim-Subscription-Key': subscriptionKey}
    conn = http.client.HTTPSConnection(host)
    query = urllib.parse.quote(search)
    conn.request("GET", path + "?q=" + query, headers=headers)
    response = conn.getresponse()
    headers = [k + ": " + v for (k, v) in response.getheaders()
                   if k.startswith("BingAPIs-") or k.startswith("X-MSEdge-")]
    return(headers, response)


def save_image(http_response, filepath='image_result.jpg'):
    http_response = http_response.read()
    http_response = http_response.decode('utf-8')
    response_dict = json.loads(http_response)
    image_url = response_dict['value'][1]['contentUrl']
    print('Saving image...')
    urllib.request.urlretrieve(image_url, filepath)



def generate_headline_document(headline_text, headline_image, filename='new_headline.pdf'):
    c = canvas.Canvas(filename)
    c.setFont('Helvetica-Bold', 20)
    c.drawString(80,750,headline_text)
    c.drawInlineImage(headline_image, x=90, y=220, preserveAspectRatio=True, width=10*cm, anchor='w')
    c.save()

    
def generate_fake_news(sentiment='positive', output_file='fake_news.pdf'):
    print("Generating " + sentiment + " headline")
    headline = get_headline(sentiment=sentiment,length=65)
    print(headline)
    print("Generating images")
    img = BingImageSearch(headline)[1]
    try:
        save_image(img, "headline_image.jpg")
    except HTTPError:
        generate_headline_document(headline_text=headline, headline_image='Profile_Picture.jpg',filename=output_file)
    print("Generating news")
    generate_headline_document(headline_text=headline, headline_image='headline_image.jpg',filename=output_file)
    print("Fake News generated at " + output_file)
    
    
print("Functions loaded")

generate_fake_news(sentiment=sentiment)