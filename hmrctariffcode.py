#!/usr/bin/env python
# coding: utf-8

# In[8]:


import requests

response = requests.get('https://www.trade-tariff.service.gov.uk/api/v2/chapters/34')


# In[9]:


response.json()


# In[54]:


response_1 = requests.get('https://www.trade-tariff.service.gov.uk/api/v2/sections')
sections = response_1.json()


# In[55]:


print(type(sections))


# In[60]:


keys_sections = sections.keys()
values_sections = sections.values()
print(type(keys_sections))
print(type(values_sections))


# In[58]:


print(sections)


# In[171]:


chapter_headers = requests.get('https://www.trade-tariff.service.gov.uk/v2/sections/16.json')
chapters_dict = chapter_headers.json()
#print(type(chapters_dict))
#print(chapters_dict)
print(chapter_headers.json())


# In[86]:


# Particular Section - Relevant Feature Extraction
# Data Structure of Section Storage - Dictionary
sect_keys = chapters_dict.keys()
print("Main Section Dictionary: ", sect_keys)
# First Key - 'data' is an inner dictionary.
data_dictionary = chapters_dict.get('data')
print("Data Key Dictionary: ", data_dictionary.keys())
print("Id: ", type(data_dictionary.get('id')))
print("Type: ", type(data_dictionary.get('type')))
print('Attributes: ', type(data_dictionary.get('attributes')))
print('Relationships: ', type(data_dictionary.get('relationships')))


# In[102]:


# Extract id, type, attributes, relationships
print(list(data_dictionary))
ids = 'Id: ' + data_dictionary.get('id')
print(ids)
types = 'Type: ' + data_dictionary.get('type')
print(types)
attributes = data_dictionary.get('attributes')
print(attributes.keys())
attribute_title = attributes.get('title')
print('Attribute Title: ', attribute_title)
attribute_chapter_to = attributes.get('chapter_to')
print('Attribute Chapter to: ', attribute_chapter_to)
attribute_chapter_from = attributes.get('chapter_from')
print('Attribute Chapter from: ', attribute_chapter_from)


# In[104]:


# Similar iteration required foron the whole dictionary, for extracting corresponding key-value pairs
for x in chapters_dict:
    print(data_dictionary[x])
    for x in attributes:
    print(attributes[x])


# In[174]:


commodity_hierarchy = requests.get('https://www.trade-tariff.service.gov.uk/v2/headings/8482.json')
commodity = commodity_hierarchy.json()
#print(type(commodity))
#print(commodity.get('included')[0])
commodity_hierarchy.json()


# In[128]:


print(commodity.keys())
print(type(commodity.get('included')))
print(type(commodity.get('included')[0]))
print(commodity.get('included')[0].keys())
#print(commodity.get('included')[0].get('attributes'))
print(commodity.get('included')[0].get('attributes').keys())
hc2_code = commodity.get('included')[0].get('attributes').get('goods_nomenclature_item_id')
print('HC_2 Code: ', hc2_code[:2])


# In[145]:


commodity_included = commodity.get('included')
for x in range(0, len(commodity_included)):
    attr = commodity_included[x].get('attributes')
    if (commodity_included[x].get('type') =='chapter'):
        hc2_code = attr.get('goods_nomenclature_item_id')
        print('HC2 Code: ', hc2_code[:2])
        description2 = attr.get('description')
        print('Description: ', description2)
        print('')
    if(attr.get('number_indents') == 1):
        hc6_code = attr.get('goods_nomenclature_item_id')
        description6 = attr.get('description')
        leaf6 = attr.get('leaf')
        print('HC6 Code: ', hc6_code[:6])
        print('Description: ', description6)
        print('Node Value: ', leaf6)
        print('')
    elif(attr.get('number_indents') == 2):
        hc_code8 = attr.get('goods_nomenclature_item_id')
        description8 = attr.get('description')
        leaf8 = attr.get('leaf')
        print('HC8 Code: ', hc_code8[:8])
        print('Description: ', description8)
        print('Node Value: ', leaf8)
        print('')
    elif(attr.get('number_indents') == 3):
        hc_code10 = attr.get('goods_nomenclature_item_id')
        description10 = attr.get('description')
        leaf10 = attr.get('leaf')
        print('HC10 Code: ', hc_code10[:10])
        print('Description: ', description10)
        print('Node Value: ', leaf10)
        print('')


# In[22]:


# Variations in results represented for both versions of the API
specific_commodity = requests.get('https://www.trade-tariff.service.gov.uk/v2/commodities/8414302010.json')
specific_commodity.json()


# In[149]:


# Spacy Named Entity Recognition - Categorisation of components of a sentence into units of measurement/cardinality 
import spacy

nlp = spacy.load("en_core_web_lg")
doc = nlp("Ball and cylindrical bearings: with an outside diameter of 28 mm or more but not more than 140 mm, with an operational thermal stress of more than 150°C at a working pressure of not more than 14 MPa, for the manufacture of machinery for the protection and control of nuclear reactors in nuclear power plants")

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)


# In[170]:


# Dependency Parsing Visualisation
from spacy import displacy

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

doc = nlp("Ball and cylindrical bearings: with an outside diameter of 28 mm or more but not more than 140 mm, with an operational thermal stress of more than 150°C at a working pressure of not more than 14 MPa, for the manufacture of machinery for the protection and control of nuclear reactors in nuclear power plants.")

# Removing stopwords results, in key comparison terms from sentences being deleted.
tokens = nlp(' '.join([token.text for token in doc if not token.is_stop]))
#print(type(tokens))
displacy.render(doc, style="dep", jupyter=True)


# In[169]:


# Approach 1 
    # BFS Search: Create an ID Guide corresponding to every level of the decision tree, comprising of salient attributes.
    # Check if given input contains valid values for each of the attributes, if so traverse to next level of the tree.

# Approach 2 
    # DFS Search: Create full description by concatenatzing descriptions at each level of the HC_Code generation.
    # Compare the given input to concatenated description and return results. 


# In[172]:


# HC_4 Code Mapping


# In[ ]:




