#!/usr/bin/env python
# coding: utf-8

# In[46]:


import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet


# In[194]:


syns = wordnet.synsets("code")
#print(type(syns))
print(syns[2].name()) 
#print(type(syns[1].name()))
print(syns[2].lemmas()[0].name()) 
print(syns[0].definition()) 
#print(type(syns[1].definition()))
print(len(syns))


# In[51]:


# Import dataset
import numpy as np
import pandas as pd
import datetime
# Cosine 
from scipy import spatial
# KNN & TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
# spacy
import spacy
nlp = spacy.load('en_core_web_lg')
# Gensim
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
# Plotting tools
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim
# Warnings
import warnings
warnings.filterwarnings('ignore')


# In[52]:


#data importing
def read_trim():
    df = pd.read_excel ('/Users/Abhi/koiosDatabase_Concepts_and_Definitions.xlsx')
    usefulAttributes = ['CONCEPT_ID', 'TERM_ID', 'DEFINITION_ID' ,'DEFINITION_CONTENT_ID','SYNONYMS_ID', 
    'CONCEPT_TYPE_ID', 'SYNONYM_VALUE', 'DEFINITION','DEF_FULL_SOURCE_TEXT','TERM_SOURCE_ID']
    for x in df.columns.values:
        if x not in usefulAttributes:
            df.drop(x, 1, inplace=True)  
    return df


# In[56]:


#tokenizing a description
#return list of "cleaned" words
def remove_punct_stop(description):
    low = [str(t) for t in nlp(description) if t.is_alpha and not t.is_stop]
    return low

#create a dictionary of terms
def term_concepts(df):
    # passed as a list of lists of words for gensim bigram
    descriptions = []
    
    keys = set(df['SYNONYM_VALUE'])
    term_to_concepts = dict.fromkeys(keys)

    count = 0
    for index, row in df.iterrows():
        if count >-1 :
            count = count +1
            term = row['SYNONYM_VALUE']
            concept_id = row['CONCEPT_ID']
            concept_type = row['CONCEPT_TYPE_ID']
            definition = row['DEFINITION']
            
            descriptions.append(remove_punct_stop(definition))

            if term_to_concepts[term] == None:
                term_to_concepts[term] = {}

            #use concept id as keys for the dictionary of each term
            term_to_concepts[term][concept_id] = (concept_type, definition)
        
    return term_to_concepts, descriptions


# In[57]:


#delete terms with only one concept
def deletekeys(term_to_concepts):
    for x in list(term_to_concepts.keys()):
        if len(term_to_concepts[x]) == 1: 
            del term_to_concepts[x]
    #return term_to_concepts


# In[55]:


#train and return the bigram generator
def trainBigram(description, min_count, threshold):
    phrases = Phrases(description, min_count = min_count, threshold = threshold)
    bigram = Phraser(phrases)
    return bigram


# In[ ]:


print("STAGE I")   
df = read_trim()
print("Term-Concept Dictionary is being loaded...")
term_to_concepts, description = term_concepts(df)
#print("Bigram model is being trained...")
#bigram = trainBigram(description,1,1)
#print("Deleting keys")
#deletekeys(term_to_concepts)


# In[198]:


# Term-based matching methods
# Case 1: When given term is found in term-concept dictionary
def check_dictionary(term, term_concept):
    keys = list(term_concept.keys())
    if term in keys:
        print("Match found")
        result = term_concept[term]
        
        return result
    else:
        print("Checking synonyms")
        check_synonym_dictionary(term, term_concept)

# Case 2: When term not found, generate synonyms and search against term-concept dictionary
# Case 3: When synonym match not found, generate corresponding synonym definition; compute similarity and if score exceeds threshold, then print corresponding concept id, type and definition
def check_synonym_dictionary(synonym, term_concept):
    nlp = spacy.load('en_core_web_lg')
    keys = term_concept.keys()
    syns = wordnet.synsets(synonym)
    
    if(len(syns)!= 0):
        for x in syns:
            if x in keys:
                print("Found", term_concept[x])
                return term_concept[x]
            else:
                print("Perform search against definitions of synonyms")
                for term in keys:
                    for concept in list(term_to_concepts[term]):
                        types, definition = term_to_concepts[term][concept]
                        #print("Dictionary Definition: ", definition)
                        #print("Synonym Definition: ", x.definition())
                        
                        syn_definition = nlp(x.definition())
                        definition_des = nlp(definition)
                        result = syn_definition.similarity(definition_des)

                        #print(result)
                        
                        if (result >= 0.85):
                            print("Success!")
                            check = term_concept[term][concept]
                            print("Synonym Definition: ", x.definition())
                            print("Results: ", concept, check)
                            print("Similarity Score", result)
                            return check
                        
                        else:
                            #print("Add new term into dictionary!")
                            break


# In[199]:


# Approach 1
#syn_def_vec = remove_punct_stop(syn_definition)
#def_vec = remove_punct_stop(syn_definition)
#text = [syn_def_vec, def_vec]
#v1 = text[0].toarray()
#v2 = text[1].toarray()
#result = 1 - spatial.distance.cosine(v1,v2)


# In[201]:


# Case 1: User gives term - Search for term in term-concept dictionary
term = 'wetting agent'
check_dictionary(term, term_to_concepts)


# In[174]:


# Case 2/3: If given term not found, generate synonyms and search in term-concept dictionary
term = 'silica'
check_dictionary(term, term_to_concepts)


# In[175]:


# Case 2/3: If given term not found, generate synonyms and search in term-concept dictionary
term = 'disinfectant'
check_dictionary(term, term_to_concepts)


# In[184]:


# Case 1: User gives term - Search for term in term-concept dictionary
term = 'leads'
check_dictionary(term, term_to_concepts)


# In[177]:


term = 'X_ray'
check_dictionary(term, term_to_concepts)


# In[203]:


term = 'acoustic'
check_dictionary(term, term_to_concepts)


# In[66]:


syn = df['SYNONYM_VALUE']
print(syn)


# In[68]:


df[df['SYNONYM_VALUE']=='wire']['DEFINITION'] 


# In[69]:


print(len(df[df['SYNONYM_VALUE']=='wire']['DEFINITION'] ))


# In[186]:


df[df['SYNONYM_VALUE']=='snow']['DEFINITION'] 


# In[73]:


for x in range (0,62123):
    print(syn[x])


# In[36]:


for x in range(0,2):
    print(df['DEFINITION'][x])


# In[ ]:


df[df['SYNONYM_VALUE']=='o-ring']['DEFINITION'] 

