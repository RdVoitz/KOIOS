#!/usr/bin/env python
# coding: utf-8

# In[2]:


from socket import *
import requests


# In[1]:


import pandas as pd
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim

df = pd.read_excel ('koiosDatabase_Concepts_and_Definitions.xlsx') #for an earlier version of Excel, you may need to use the file extension of 'xls'
#print(df.columns.values)


# In[2]:


values = []
repeated = []
count = 0
for x in df['CONCEPT_ID']:
    if x in values and x not in repeated:
        count = count + 1
        repeated.append(x)
    else:
        values.append(x)
print(count)


# In[3]:


usefulAttributes = ['CONCEPT_ID', 'TERM_ID', 'DEFINITION_ID' ,'DEFINITION_CONTENT_ID','SYNONYMS_ID', 
'CONCEPT_TYPE_ID', 'SYNONYM_VALUE', 'DEFINITION','DEF_FULL_SOURCE_TEXT','TERM_SOURCE_ID']

for x in df.columns.values:
    if x not in usefulAttributes:
        df.drop(x, 1, inplace=True)       
#print(df.columns.values)


# In[4]:


#create a dictionary of terms

keys = set(df['SYNONYM_VALUE'])
term_to_concepts = dict.fromkeys(keys)

for index, row in df.iterrows():
    term = row['SYNONYM_VALUE']
    concept_id = row['CONCEPT_ID']
    concept_type = row['CONCEPT_TYPE_ID']
    definition = row['DEFINITION']
    
    if term_to_concepts[term] == None:
        term_to_concepts[term] = {}
    
    #use concept id as keys for the dictionary of each term
    term_to_concepts[term][concept_id] = (concept_type, definition)
    


# In[5]:


#delete terms with only one concept
def deletekeys():
    for x in list(term_to_concepts.keys()):
        if len(term_to_concepts[x]) == 1:
            del term_to_concepts[x]

deletekeys()
print(len(term_to_concepts))


# In[6]:


import spacy
nlp = spacy.load('en_core_web_lg')


# In[7]:


#Compares two descriptions using the trained tfidf
#Returns similarity score measured with cosine
nlp = spacy.load('en_core_web_lg')
def compare(d1,d2, tfidf):
    doc1 = nlp(d1)
    doc2 = nlp(d2)
     
    #Clean the text
    doc1 = ' '.join([str(t) for t in doc1 if not t.is_stop | t.is_punct ])
    doc2 = ' '.join([str(t) for t in doc2 if not t.is_stop | t.is_punct ])
    
    text = [doc1, doc2]
    
    #get the word vector for each description to be compared
    v1 = tfidf.transform([text[0]]).toarray()
    v2 = tfidf.transform([text[1]]).toarray()
    
    result = 1 - spatial.distance.cosine(v1,v2)

    #result = doc1.similarity(doc2)  #Using spacy built in similarity
    return result


# In[8]:


# import gensim.downloader as api

# dataset = api.load("text8")
# data = [d for d in dataset]

# def create_tagged_document(list_of_list_of_words):
#     for i, list_of_words in enumerate(list_of_list_of_words):
#         yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])
        

# train_data = list(create_tagged_document(data))

# model = gensim.models.doc2vec.Doc2Vec(vector_size = 50, min_count = 2, epochs = 1)
# model.build_vocab(train_data)

# model.train(train_data, total_examples= model.corpus_count, epochs = model.epochs)


# In[8]:


#SAME methods for cleaning the text and tokenizing
import logging
import numpy as np
from gensim.models import word2vec
import wikipedia
from wikipedia import search, page


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)

def description_to_wordlist(description):
    d  = [str(t).lower() for t in nlp(description) if not t.is_stop] # | t.is_punct ]
    return d

def wikipedia_summary(term):
    text = []
    titles, suggestions  = search(term, suggestion = True)
    if suggestions == None:
        x = 0
        while x<3:
            x = x+1
           # print(term, 'X is ', x, 'Len_titles is ', len(titles))
            try:
                summary = wikipedia.summary(titles[x-1])
                text.append(description_to_wordlist(summary))
            except wikipedia.DisambiguationError as e:
                for y in e.options:
                    try:
                        summary = wikipedia.summary(y)
                        text.append(description_to_wordlist(summary))
                    except:
                        continue
            except:
                continue
            #x = x+1
    else:
        try:
            summary = wikipedia.summary(suggestions)
            text.append(description_to_wordlist(summary))
        except wikipedia.DisambiguationError as e:
            for y in e.options:
                    try:
                        summary = wikipedia.summary(y)
                        text.append(description_to_wordlist(summary))
                    except:
                        continue
        except:
            return text
            
    return text

def create_text():
    count = 0
    text=[]
    print("Preparing the text for the model")
    for term in list(term_to_concepts.keys()):
        count = count +1
        if count % 100 == 0 :
            print("100 TERMS COVERED")
        wiki = wikipedia_summary(term)
        text = text + wiki
            
        for concept in list(term_to_concepts[term]):
            t, d = term_to_concepts[term][concept]
            #print(d)
            d = description_to_wordlist(d)
            text.append(d)  
    return text

def train_GENSIM(text):

    #traing GENSIM WORD2VEC

    num_features = 300    # Word vector dimensionality                      
    min_word_count = 40   # Minimum word count                        
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size                                                                                    
    downsampling = 1e-3   # Downsample setting for frequent words
    epochs = 10           # Number of epochs


    print("Training model...")
    model = word2vec.Word2Vec(text,iter=epochs, workers=num_workers,                 size=num_features, min_count = min_word_count,                 window = context, sample = downsampling)
    
    model_name = "300features_40minwords_10context"
    model.save(model_name)
    
    return model


def vector_averaging(doc, model):
    vector = np.zeros((300,),dtype="float32")
    index2word_set = set(model.wv.index2word)
    nwords = 0
    
    for word in doc:
        if word in index2word_set:
            nwords = nwords + 1
            vector = np.add(vector, model[word])
    vector = np.divide(vector,nwords)
    
    return vector

def gensim_similarity(d1,d2, model):
    #Clean the text
    doc1 = [str(t) for t in nlp(d1) if not t.is_stop | t.is_punct ]
    doc2 = [str(t) for t in nlp(d2) if not t.is_stop | t.is_punct ]
        
    v1 = vector_averaging(doc1,model)
    v2 = vector_averaging(doc2,model)

    result = 1 - spatial.distance.cosine(v1,v2)
    return result


# In[31]:


import wikipedia
from wikipedia import search, page
title = search('wire',suggestion = True)

print(title)
wikipage = page(title[0])

print(type(wikipedia.summary(title[0])))


# In[18]:


# example1 = 'Sachin is a cricket player and a opening batsman'
# example2 = 'Dhoni is a cricket player too He is a batsman and keeper'
# example3 = 'Anand is a chess player'
# example4 = 'This is such a sunny day'

# print("E1 & E2 ", compare(example1,example2))
# print("E1 & E3 ", compare(example1,example3))
# print("E1 & E4 ", compare(example1,example4))
# print("E2 & E3 ", compare(example2,example3))
# print("E2 & E4 ", compare(example2,example4))
# print("E3 & E4 ", compare(example3,example4))


# In[9]:


# method for training the tf-idf on all of the descriptions
def trainer():
    text = []
    for term in list(term_to_concepts.keys()):
        for concept in list(term_to_concepts[term]):
            t, d = term_to_concepts[term][concept]
            d = ' '.join([str(t) for t in nlp(d) if not t.is_stop | t.is_punct ])
            text.append(d)
    tfidf = TfidfVectorizer(ngram_range = (1,3))
    tfidf.fit(text)
    return tfidf

def trainer_w_term(term):
    text = []
    for concept in list(term_to_concepts[term]):
            t, d = term_to_concepts[term][concept]
            d = ' '.join([str(t) for t in nlp(d) if not t.is_stop | t.is_punct ])
            text.append(d)
    tfidf = TfidfVectorizer(ngram_range = (1,2))
    tfidf.fit(text)
    return tfidf   
    


# In[10]:


import datetime
from gensim.models import Word2Vec

print(datetime.datetime.now())

#list of terms used as keys
doublekeys = list(term_to_concepts.keys()) #list of concepts

# list of touples made of similar descriptions
similar = [] 

#dictionary of
type_to_similar = dict.fromkeys(set(doublekeys))

#trained tfidf model
tfidf = trainer()  

#PREPARE TEXT DATA FOR GENSIM TRAINING
#text = create_text()

#GENSIM WORD2VECTOR TRAINING
#model = train_GENSIM(text)

#retreive already trained model
# model = Word2Vec.load("300features_40minwords_10context")
# count = 0

for term in doublekeys:
    #while count<100:
    count = count + 1

    smalkeys = list(term_to_concepts[term].keys()) #list of concepts associated with a term

    #compare pairs of two descriptions by navigating the dictionary
    #of each term and addint the pairs with a level of similarity above a certain threshold
    #in a new dictionary associated with the term

    #tfidf = trainer_w_term(term)
    for i in range(0, len(smalkeys)-1):
        conceptid1 = smalkeys[i]
        typeid1, description1 = term_to_concepts[term][conceptid1]

        for j in range(i+1, len(smalkeys)):
            conceptid2 = smalkeys[j]
            typeid2, description2 = term_to_concepts[term][conceptid2]

            #the type of the concepts should be the same for comparison

            if typeid1 == typeid2:
                if compare(description1,description2,tfidf) >= 0.80: #gensim_similarity(description1,description2,model)>=0.90: 
                    similar.append((description1,description2)) #add pairs to a list just for keeping track 
                    if type_to_similar[term] == None:
                        type_to_similar[term] = []
                    type_to_similar[term] = type_to_similar[term] + [(description1,description2)]
                                  
print(datetime.datetime.now())        


# In[32]:


print(len(similar))

print(len(type_to_similar))

print(len(type_to_similar.keys()))


# In[36]:


def mostSimilar():
    m = 0
    memorised = None
    descriptions = {}
    count = 0
    
    for term in set(doublekeys):
        if not type_to_similar[term] == None: 
            count = count +1
            if len(type_to_similar[term]) > m:
                memorised = term
                m = len(type_to_similar[term])
    print("Maximum number of similar descriptions for a term  ",m)
    print("The term for which the max was found  ", memorised)
    print(count)
    print('**************************')
    
    memorised = "standard form"
    
    for touple in type_to_similar[memorised]:
        d1,d2 = touple
        
        if not d1 in list(descriptions.keys()):
            descriptions[d1] = [d1,d2]
        else:
            descriptions[d1] = descriptions[d1] + [d2]
    
    m = 0
    memorised = None
    for d in list(descriptions.keys()):
        if len(set(descriptions[d])) > m:
            m = len(set(descriptions[d]))
            memorised = d
    
    print(len(set(descriptions[memorised])))
    print(memorised)
    print('**************************')
    
    for d in set(descriptions[memorised]):
        print(d)
        
mostSimilar()    


# In[33]:


model.most_similar("wire")


# In[ ]:





# In[57]:


#GENSIM METHOD
from gensim import corpora
from gensim.models import TfidfModel
import numpy as np
import pprint

def create_dictionary():
    text = []
    for term in list(term_to_concepts.keys()):
            for concept in list(term_to_concepts[term]):
                t, d = term_to_concepts[term][concept]
                d = [str(t).lower() for t in nlp(d) if not t.is_stop | t.is_punct ]
                text.append(d)
                #d = [d]
            break
    dictionary = corpora.Dictionary(text)
    corpus = [dictionary.doc2bow(doc, allow_update = True) for doc in text]
    
#     tfidf = TfidfModel(corpus, smartirs='ntc')
#     for doc in tfidf[corpus]:
#         print([[dictionary[ids], np.around(freq, decimals=3)] for ids, freq in doc])
        
#create_dictionary()

