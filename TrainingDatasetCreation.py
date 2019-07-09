#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load XML dataset into dataframe
import xml.etree.ElementTree as ET
import pandas as pd
import urllib.request

url = "https://opendata-api.iec.ch/v1/opendata/iev/112/00588d84-e45a-442d-ad6a-09a9215ac6aa"

def parse_xml(url):
    tree = ET.parse(urllib.request.urlopen(url))
    tree_root = tree.getroot()
    return tree_root 

root = parse_xml(url)


# In[3]:


# Create list of attributes based on parent xml tags
def create_list_attrib(root, attribute_name):
    attribute_list = []
    for node in root.iter(attribute_name):
        attribute = node.attrib
        attribute_list.append(attribute)
    return attribute_list

for x in create_list_attrib(root,'lang-set'):
    print(x)


# In[4]:


# Create list of attributes based on values of xml tags
def create_list_text(root, attribute_name):
    attribute_list = []
    for node in root.iter(attribute_name):
        attribute = node.text
        attribute_list.append(attribute)
    return attribute_list

for x in create_list_text(root,'definition'):
    print(x)


# In[5]:


attributes = []
attributes = [create_list_attrib(root,'concept'), create_list_text(root,'term-name'), 
              create_list_text(root, 'definition'), create_list_attrib(root,'lang-set')]
print(attributes)


# In[7]:


def create_dataframe(attributes):
    dataframe = {'concept ievref' : attributes[0], 'term-name' : attributes[1], 'definition' : attributes[2], 'language id' : attributes[3]}
    return pd.DataFrame(dataframe)   


# In[9]:


df = create_dataframe(attributes)
print(df)


# In[10]:


df.head()


# In[11]:


df.shape


# In[12]:


# Number of null values
len(df) - df.count()


# In[ ]:




