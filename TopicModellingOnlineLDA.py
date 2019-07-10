#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Import dataset
import pandas as pd
import datetime

# spacy
import spacy

# Gensim
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

# Text Preprocessing
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Plotting tools
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim


# In[3]:


def read_trim():
    df = pd.read_excel ('koiosDatabase_Concepts_and_Definitions.xlsx')
    usefulAttributes = ['CONCEPT_ID', 'TERM_ID', 'DEFINITION_ID' ,'DEFINITION_CONTENT_ID','SYNONYMS_ID', 
    'CONCEPT_TYPE_ID', 'SYNONYM_VALUE', 'DEFINITION','DEF_FULL_SOURCE_TEXT','TERM_SOURCE_ID']
    for x in df.columns.values:
        if x not in usefulAttributes:
            df.drop(x, 1, inplace=True)  
    return df


# In[4]:


def term_to_concepts(df):
    
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

            if term_to_concepts[term] == None:
                term_to_concepts[term] = {}

            #use concept id as keys for the dictionary of each term
            term_to_concepts[term][concept_id] = (concept_type, definition)
        
    return term_to_concepts


# In[5]:


def deletekeys(term_to_concepts):
    for x in list(term_to_concepts.keys()):
        if len(term_to_concepts[x]) == 1: #len(
            del term_to_concepts[x]
    print(len(term_to_concepts))
    return term_to_concepts


# In[6]:


dataframe = read_trim()
dataframe.head()


# In[7]:


dataframe.shape


# In[8]:


term_concept = term_to_concepts(dataframe)
print(len(term_concept))
new_term_concept = deletekeys(term_concept)
print(new_term_concept)


# In[9]:


#Topic Modelling on descriptions of concepts in the KOIOS Dictionary - multiple descriptions mapping to single term
def descriptions(new_term_concept):
    description_list = []
    for x in list(new_term_concept.keys()):
        for y in list(new_term_concept[x]):
            concept, description = new_term_concept[x][y]
            description_list.append(description)
    return description_list
test_descriptions = descriptions(new_term_concept)    


# In[10]:


# Break down sentence into list of words
def tokenize(sentences):
    tokenizer = RegexpTokenizer(r'\w+')
    tokenize_data = []
    for sentence in sentences:
        # Convert to lower case
        raw_data = sentence.lower()
        token_words = tokenizer.tokenize(raw_data)
        tokenize_data.append(token_words)
    return tokenize_data

tokenize_words = tokenize(test_descriptions)
print(tokenize_words)


# In[11]:


# Build bigram and trigram models
# Higher the threshold, lower the phrases
bigram = gensim.models.Phrases(tokenize_words, min_count=1, threshold=1)
trigram = gensim.models.Phrases(bigram[tokenize_words], threshold=1)

# Quicker to extract bigrams/trigrams out of sentences
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# In[12]:


# Remove stopwords
def remove_stopwords(des_sentences):
    stop_words = set(stopwords.words('english'))
    return [[text for text in des if not text in stop_words] for des in des_sentences]

removeStopwords = remove_stopwords(tokenize_words)
print(removeStopwords)


# In[13]:


# Create bigrams
def make_bigrams(des_sentences):
    return [bigram_mod[des] for des in des_sentences]

bigrams = make_bigrams(removeStopwords)
print(bigrams)


# In[14]:


# Create trigrams
def make_trigrams(des_sentences):
    return[trigram_mod[bigram_mod[des]] for des in des_sentences]


# In[15]:


def lemmatization(texts):
    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
    stemmingList = []
    for text in texts:
        doc = nlp(" ".join(text))
        stemmingList.append([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags])
    return stemmingList

nlp = spacy.load('en', disable=['parser', 'ner'])

# Perform lemmatisation, and only keep Noun, Adj, Verb, Adverb
data_lemmatized = lemmatization(bigrams)

print(data_lemmatized[:2])


# In[16]:


def prepare_corpus(preprocessedDescription):
    # Create Dictionary
    dictionary = corpora.Dictionary(preprocessedDescription)
    
    # Create corpus
    texts = preprocessedDescription
    
    # Term Document Frequency
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    return dictionary,corpus

dictionary, corpus = prepare_corpus(data_lemmatized)
print(corpus[:1])


# In[17]:


print(dictionary[0])


# In[18]:


# Human readable format of corpus (term-frequency)
[[(dictionary[id], freq) for id, freq in corp] for corp in corpus[:1]]


# In[23]:


# Buid the LDA - Online Learning Topic Model
def create_lda_model(preprocessedDescription):
    dictionary, corpus = prepare_corpus(preprocessedDescription)

    # online LDA training - processes the whole corpus in one pass, then updates the model, then another pass and so on.
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
    
    print(lda_model.print_topics())
    #print(lda_model.top_topics(document_term_matrix, dictionary, coherence = 'c_v', topn = words))
    
    #for i, topic in lda_model.show_topics(formatted=True, num_topics = 10, num_words = 5):
        #print(str(i) +": "+ topic)
        #print()
        
    #print("Perplexity score: ", lda_model.log_perplexity(document_term_matrix))
    return lda_model

lda_model = create_lda_model(data_lemmatized)


# In[24]:


# Computing Perplexity - A measure of how good the model is, the lower the better.
print('\n Perplexity: ', lda_model.log_perplexity(corpus))

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\n Coherence Score: ', coherence_lda)


# In[25]:


# Visualise the topics
lda_model = create_lda_model(data_lemmatized)
dictionary,corpus = prepare_corpus(data_lemmatized)

pyLDAvis.enable_notebook()
#print("Started", datetime.datetime.now())
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
vis
#print("Ended", datetime.datetime.now())


# In[26]:


# This method trains multiple LDA models and provides the models and their corresponding coherence scores
def compute_coherence_values(dictionary, corpus, preprocessedDescriptions, limit, start=2, step=3):
    # dictionary: Gensim dictionary
    # corpus: Gensim corpus
    # preprocessedDescriptions: list of processed descriptions
    # limit: maximum number of topics
    
    coherence_values = []
    model_list = []
    
    for numberTopics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus, num_topics = numberTopics, id2word = dictionary, passes = 1)
        model_list.append(model)
        coherence_model = CoherenceModel(model = model, texts = preprocessedDescriptions, dictionary = dictionary, coherence = 'c_v')
        coherence_values.append(coherence_model.get_coherence())
        
    return model_list, coherence_values


# In[28]:


# Plot the coherence score values
def plot_graph(preprocessedDescription, start, limit, step):
    dictionary, corpus = prepare_corpus(preprocessedDescription)
    model_list, coherence_values = compute_coherence_values(dictionary, corpus, preprocessedDescription, limit, start, step)
    
    x = range(start, limit, step)
    plt.plot(x,coherence_values)
    #plt.xticks(x)
    plt.xlabel("Number of topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc = 'best')
    plt.show()
    
    for m, cv in zip(x, coherence_values):
        print("Number of Topics =", m, " has Coherence Value of", round(cv, 4))


# In[30]:


start,limit,step = 2,50,1
plot_graph(data_lemmatized, start, limit, step)


# In[ ]:




