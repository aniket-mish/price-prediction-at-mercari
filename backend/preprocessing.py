import numpy as np
import string
import re
import pandas as pd
from sklearn import preprocessing

#stopwords without no, not, etc
STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]


def handle_missing_values(data):
    
    data.fillna({'name':'missing', 'item_description':'missing'}, inplace=True)
    
    return data


def remove_emoji(sentence):
    
    pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    return pattern.sub(r'', sentence)


def remove_punctuation(sentence):
    
    regular_punct = list(string.punctuation)
    
    for punc in regular_punct:
        if punc in sentence:
            sentence = sentence.replace(punc, ' ')

    return sentence.strip()


def decontracted(phrase):
    
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    
    return phrase


def process_text(data, cols):
    
    for col in cols:
        
        processed_data = []
        
        for sentence in data[col].values:
            
            sent = decontracted(sentence)
            sent = sentence
            sent = sent.replace('\\r', ' ')
            sent = sent.replace('\\"', ' ')
            sent = sent.replace('\\n', ' ')
            sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
            sent = remove_emoji(sent)
            sent = remove_punctuation(sent)
            sent = ' '.join(e for e in sent.split() if e not in STOPWORDS)
            processed_data.append(sent.lower().strip())
            
        data[col] = processed_data
        
    return data


def process_category(data):
    
    for i in range(3):
        
        def get_part(x):
            
            if type(x) != str:
                return np.nan
        
            parts = x.split('/')
            
            if i >= len(parts):
                return np.nan
            else:
                return parts[i]

        field_name = 'category_' + str(i)
        
        data[field_name] = data['category_name'].apply(get_part)
    
    return data

def get_features(data):
    
    luxury_brands = ["MCM", "MCM Worldwide", "Louis Vuitton", "Burberry", "Burberry London", "Burberry Brit", "HERMES", "Tieks",
                     "Rolex", "Apple", "Gucci", "Valentino", "Valentino Garavani", "RED Valentino", "Cartier", "Christian Louboutin",
                     "Yves Saint Laurent", "Saint Laurent", "YSL Yves Saint Laurent", "Georgio Armani", "Armani Collezioni", "Emporio Armani"]
    
    data['is_luxurious'] = (data['brand_name'].isin(luxury_brands)).astype(np.int8)

    expensive_brands = ["Michael Kors", "Louis Vuitton", "Lululemon", "LuLaRoe", "Kendra Scott", "Tory Burch", "Apple", "Kate Spade",
                  "UGG Australia", "Coach", "Gucci", "Rae Dunn", "Tiffany & Co.", "Rock Revival", "Adidas", "Beats", "Burberry",
                  "Christian Louboutin", "David Yurman", "Ray-Ban", "Chanel"]

    data['is_expensive'] = (data['brand_name'].isin(expensive_brands)).astype(np.int8)
    
    return data
    

def preprocess(data):

    #handle missing values
    data = handle_missing_values(data)
    
    data = process_category(data)
    
    data = process_text(data, ['name', 'item_description', 'category_name'])
    
    data = get_features(data)
    
    data.fillna({'brand_name': ' ', 'category_0': 'other', 'category_1': 'other', 'category_2': 'other'}, inplace = True)
    
    #concat columns
    data['name'] = data['name'] + ' ' + data['brand_name'] + ' ' + data['category_name'] 
    data['text'] = data['name'] + ' ' + data['item_description']
    
    #drop columns which are not required
    data = data.drop(columns = ['brand_name', 'item_description', 'category_name'], axis = 1)

    return data