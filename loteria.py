# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 20:17:50 2017

@author: iv
"""
import numpy as np
import pandas as pd
import re
import lxml
import requests as rq
import urllib
import tables as tb
import bs4
import shutil
import io
import curl
import pyPdf
import os
import sklearn
import scipy
#import pdftables_api

## Parte 1 - Obtencion lista con las url de los ficheros .pdf con los resultados de los sorteos

url = "http://www.laprimitiva.info/historico/listado.html"
print("")

suffix = ".pdf"
link_list = []

def getPDFs():    
    # Gets URL from user to scrape
    response = rq.get(url, stream=True)
    soup = bs4.BeautifulSoup(response.text,'lxml')

    for link in soup.find_all('a'): # Finds all links
        if suffix in str(link): # If the link ends in .pdf
            link_list.append(link.get('href'))
    print(link_list)

getPDFs()
linklist = ["http://www.laprimitiva.info/" + str.split(x,'../')[1] for x in link_list]

### Parte 2

url = "http://www.lotoideas.com/primitiva-resultados-historicos-de-todos-los-sorteos/"
print("")

suffix = "=csv"
link_list = []

def getPDFs():    
    # Gets URL from user to scrape
    response = rq.get(url, stream=True)
    soup = bs4.BeautifulSoup(response.text,'lxml')

    for link in soup.find_all('a'): # Finds all links
        if suffix in str(link): # If the link ends in .pdf
            link_list.append(link.get('href'))
    print(link_list)

getPDFs()

primitiva1_csv = urllib.urlopen(link_list[0]) 
primitiva2_csv = urllib.urlopen(link_list[1]) 

primitiva1 = pd.read_csv(primitiva1_csv, index_col=0, parse_dates=True)
primitiva1.columns = ['bola_1','bola_2','bola_3','bola_4','bola_5','bola_6','comp', 'joker' ]
primitiva2 = pd.read_csv(primitiva2_csv, index_col=0, parse_dates=True)
primitiva2.columns = ['bola_1','bola_2','bola_3','bola_4','bola_5','bola_6','comp', 'joker' ]

primitiva = pd.concat([primitiva2, primitiva1], ignore_index=True).sort_index(ascending = False )
primitiva = pd.concat([primitiva2, primitiva1], ignore_index=False).sort_index(ascending = False )

primitiva.hist()
primitiva_2 = primitiva.transpose()
primi_days = primitiva_2.iloc[0:7,:]
primi_balls = primitiva.iloc[:,0:7]
primi_days.to_csv('hoy.csv')
primitiva.to_csv('manana.csv')

from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
sklearn.cro

lr = linear_model.LinearRegression()
#boston = datasets.load_boston()
#y = boston.target

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, primi_balls, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


model = sklearn.semi_supervised
primi = primitiva_2.iloc[0:6,:]
model.fit_transform(x for x in primitiva_2.columns)
