#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

dataset = pd.read_excel('koiosDatabase_Concepts_and_Definitions.xlsx')


# In[2]:


import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer


# In[3]:


data = dataset['DEFINITION']


# In[4]:


print(len(data))


# In[5]:


def simplified_description():
    simplified_version = []
    for x in range(0,1000):
        #print(data[x])
        simplified_version.append(data[x])
    return simplified_version

simplified_description()


# In[6]:


print(type(simplified_description()))
#print(len(simplified_version))

#from sklearn.feature_extraction.text import TfidfVectorizer

#data = simplified_description()
#transformer = TfidfVectorizer()
#tfidf = transformer.fit_transform(data)
    


# In[11]:


#from sklearn.feature_extraction.text import TfidfVectorizer
from string import *
def clean_data(simpDescriptions):
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    porter_stemmer = PorterStemmer()
    description_texts = []
    
    for des in simpDescriptions:
        #description_texts.append(simpDescriptions[des])
        #print(type(description_texts[des]))
        #print(type(simpDescriptions))
        #description_texts[des] = description_texts[des].lower()
        #description_texts[des] = tokenizer(description_texts[des])
        #description_texts[des] = [word for word in description_texts[des] if not word.isnumeric()]
        #description_texts[des] = [stemmer.stem(word) for word in description_texts[des] if not word in stop_words]
        #description_texts[des] = join(description_texts[des]," ")
        raw_data = des.lower()
        tokens_words = tokenizer.tokenize(des)
        remove_stopwords = [text for text in tokens_words if not text in stop_words]
        remove_digits = [text for text in remove_stopwords if not text.isnumeric()]
        remove_single_letters = [text for text in remove_digits if len(text) >= 2]
        #stemmed_tokens = [porter_stemmer.stem(text) for text in remove_single_letters]
        #description_texts.append(stemmed_tokens)
        description_texts.append(remove_single_letters)
        
    #transformer = TfidfVectorizer()
    #tfidf = transformer.fit(description_texts)
    
    return description_texts


# In[12]:


clean_data(simplified_description())


# In[13]:


def prepare_corpus(preprocessedDescription):
    dictionary = corpora.Dictionary(preprocessedDescription)
    # Filtering out words that occur less than 20 documents, or more than 50% of the documents - Check this out(maybe using a different combination of numbers)
    # dictionary.filter_extremes(no_below = 10)
    document_term_matrix = [dictionary.doc2bow(description) for description in preprocessedDescription]
    
    #print(dictionary)
    #print(len(document_term_matrix))
    
    return dictionary,document_term_matrix


# In[14]:


prepare_corpus(clean_data(simplified_description()))


# In[15]:


def create_lsa_model(preprocessedDescription, numberTopics, words):
    dictionary, document_term_matrix = prepare_corpus(preprocessedDescription)
    lsa_model = LsiModel(document_term_matrix, num_topics = numberTopics, id2word = dictionary)
    
    #print(lsa_model.print_topics(num_topics = numberTopics, num_words = words))
    
    for id, topic in lsa_model.show_topics(formatted = True, num_topics = numberTopics, num_words = words):
        print(str(id)+": "+topic)
        print()
    return lsa_model


# In[16]:


def compute_coherence_values_lsa(dictionary, document_term_matrix, preprocessedDescription, stop, start=2, step=10):
    #start - starting document/description number
    #stop - maximum number of descriptions
    #step - increment value
    coherence_values = []
    model_list = []
    
    for numberTopics in range(start, stop, step):
        model = LsiModel(document_term_matrix, num_topics = numberTopics, id2word = dictionary)
        model_list.append(model)
        coherence_model = CoherenceModel(model = model, texts = preprocessedDescription, dictionary = dictionary, coherence = 'c_v')
        coherence_values.append(coherence_model.get_coherence())
        
    return model_list, coherence_values


# In[17]:


def plot_graph(preprocessedDescription, start, stop, step):
    dictionary, document_term_matrix = prepare_corpus(preprocessedDescription)
    model_list, coherence_values = compute_coherence_values_lsa(dictionary, document_term_matrix, preprocessedDescription, stop, start, step)
    
    x = range(start, stop, step)
    plt.plot(x,coherence_values)
    plt.xticks(x)
    plt.xlabel("Number of topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc = 'best')
    plt.show()


# In[19]:


start,stop,step = 2,100,10
plot_graph(clean_data(simplified_description()), start, stop, step)


# In[20]:


numberTopics = 12
words = 5
simpDescription = simplified_description()
preprocessed_text = clean_data(simpDescription)
model = create_lsa_model(preprocessed_text, numberTopics, words)


# In[21]:


# LDA - Online Learning
from gensim.models import LdaModel
def create_lda_model(preprocessedDescription):
    dictionary, document_term_matrix = prepare_corpus(preprocessedDescription)

    # online LDA training - processes the whole corpus in one pass, then updates the model, then another pass and so on.
    lda_model = LdaModel(corpus=document_term_matrix, id2word=dictionary, num_topics=10, update_every=1, passes=1)
    
    #print(lda_model.print_topics(num_topics = numberTopics, num_words = words))
    #print(lda_model.top_topics(document_term_matrix, dictionary, coherence = 'c_v', topn = words))
    
    for i, topic in lda_model.show_topics(formatted=True, num_topics = 10, num_words = 5):
        print(str(i) +": "+ topic)
        print()
        
    print("Perplexity score: ", lda_model.log_perplexity(document_term_matrix))
    
    return lda_model


# In[22]:


simpDescription = simplified_description()
preprocessed_text = clean_data(simpDescription)
dictionary, document_term_matrix = prepare_corpus(preprocessed_text)
model = create_lda_model(preprocessed_text)


# In[23]:


import pyLDAvis
import pyLDAvis.gensim as gensimvis

pyLDAvis.enable_notebook()
visualisation = gensimvis.prepare(model, document_term_matrix, dictionary)
visualisation


# In[32]:


# LDA - Batch Learning
from gensim.models import LdaModel
def create_lda_model(preprocessedDescription, numberTopics, words):
    dictionary, document_term_matrix = prepare_corpus(preprocessedDescription)
    
    # batch LDA training - takes a chunk of documents, updates the model, takes another chunk and so on.
    lda_model = LdaModel(document_term_matrix, num_topics = numberTopics, id2word = dictionary, passes = 10, alpha = 'auto', eta = 'auto')
   
    #print(lda_model.print_topics(num_topics = numberTopics, num_words = words))
    #print(lda_model.top_topics(document_term_matrix, dictionary, coherence = 'c_v', topn = words))
    
    for i, topic in lda_model.show_topics(formatted=True, num_topics = 10, num_words = 5):
        print(str(i) +": "+ topic)
        print()
        
    print("Perplexity score: ", lda_model.log_perplexity(document_term_matrix))
    
    return lda_model


# In[33]:


def compute_coherence_values_lda(dictionary, document_term_matrix, preprocessedDescription, stop, start=2, step=5):
    #start - starting document/description number
    #stop - maximum number of descriptions
    #step - increment value
    coherence_values = []
    model_list = []
    
    for numberTopics in range(start, stop, step):
        model = LdaModel(document_term_matrix, num_topics = numberTopics, id2word = dictionary, passes = 1)
        model_list.append(model)
        coherence_model = CoherenceModel(model = model, texts = preprocessedDescription, dictionary = dictionary, coherence = 'c_v')
        coherence_values.append(coherence_model.get_coherence())
        
    return model_list, coherence_values


# In[34]:


def plot_graph(preprocessedDescription, start, stop, step):
    dictionary, document_term_matrix = prepare_corpus(preprocessedDescription)
    model_list, coherence_values = compute_coherence_values_lda(dictionary, document_term_matrix, preprocessedDescription, stop, start, step)
    
    x = range(start, stop, step)
    plt.plot(x,coherence_values)
    plt.xticks(x)
    plt.xlabel("Number of topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc = 'best')
    plt.show()


# In[40]:


start,stop,step = 2,50,5
plot_graph(clean_data(simplified_description()), start, stop, step)


# In[37]:


numberTopics = 7
words = 5
simpDescription = simplified_description()
preprocessed_text = clean_data(simpDescription)
model = create_lda_model(preprocessed_text, numberTopics, words)


# In[39]:


import pyLDAvis
import pyLDAvis.gensim as gensimvis

numberTopics = 7
words = 5
simpDescription = simplified_description()
preprocessed_text = clean_data(simpDescription)
dictionary, document_term_matrix = prepare_corpus(preprocessed_text)
model = create_lda_model(preprocessed_text, numberTopics, words)
pyLDAvis.enable_notebook()
visualisation = gensimvis.prepare(model, document_term_matrix, dictionary)
visualisation


# In[ ]:




