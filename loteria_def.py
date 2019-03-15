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
import os
import sklearn
import scipy
from itertools import combinations
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
#import pyPdf
#import pdftables_api 

####################
### PRIMITIVA ###
####################
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

primitiva1_csv = urllib.request.urlopen(link_list[0]) 
primitiva2_csv = urllib.request.urlopen(link_list[1]) 

primitiva1 = pd.read_csv(primitiva1_csv, index_col=0, parse_dates=True)
primitiva1.columns = ['bola_1','bola_2','bola_3','bola_4','bola_5','bola_6','complementario', 'reintegro', 'joker' ]
primitiva2 = pd.read_csv(primitiva2_csv, index_col=0, parse_dates=True)
primitiva2.columns = ['bola_1','bola_2','bola_3','bola_4','bola_5','bola_6','complementario', 'reintegro', 'joker' ]

primitiva = pd.concat([primitiva2, primitiva1], ignore_index=True).sort_index(ascending = False )
primitiva = pd.concat([primitiva2, primitiva1], ignore_index=False).sort_index(ascending = False )

primitiva2 = primitiva.iloc[:,0:7]
primitiva3 =  primitiva2.reset_index()
primitiva3 = primitiva3.iloc[:,1:8]
primitiva4 = primitiva3.astype(str)
#primitiva5 = primitiva4.iloc[0:100,:]
primitiva6 = primitiva4.values.tolist()
################################################################
# bolas y complementario
te = TransactionEncoder()
te_ary = te.fit(primitiva6).transform(primitiva6)
df = pd.DataFrame(te_ary, columns=te.columns_)
df
frequent_itemsets = apriori(df, min_support=0.001, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

frequent_itemsets[ (frequent_itemsets['length'] == 2) &
                   (frequent_itemsets['support'] >= 0.023) ] #Para parejas poner lengh en 2 y support mayor que 0.023, y para trios poner length en3 y support inferior a 0.005
################################################################
################################################################



####################
### EUROMILLON ###
####################
url = "http://www.lotoideas.com/euromillones-resultados-historicos-de-todos-los-sorteos/"
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

euromillones_csv = urllib.request.urlopen(link_list[0]) 

euromillones1 = pd.read_csv(euromillones_csv, index_col=0, parse_dates=True)
euromillones1.columns = ['bola_1','bola_2','bola_3','bola_4','bola_5','blanco','estrella1','estrella2']

euromillones2 = euromillones1.iloc[:,np.r_[0:5]]
euromillones2 =  euromillones2.reset_index()

estrellas2 = euromillones1.iloc[:,np.r_[6,7]]
estrellas2 =  estrellas2.reset_index()

euromillones3 = euromillones2.iloc[:,1:6]
estrellas3 = estrellas2.iloc[:,1:3]

euromillones4 = euromillones3.astype(str)
estrellas4 = estrellas3.astype(str)

euromillones5 = euromillones4.values.tolist()
estrellas5 = estrellas4.values.tolist()
################################################################
# bolas
te1 = TransactionEncoder()
te_ary1 = te1.fit(euromillones5).transform(euromillones5)
df1 = pd.DataFrame(te_ary1, columns=te1.columns_)
df1
frequent_itemsets1 = apriori(df1, min_support=0.001, use_colnames=True)
frequent_itemsets1['length'] = frequent_itemsets1['itemsets'].apply(lambda x: len(x))

frequent_itemsets1[ (frequent_itemsets1['length'] == 2) &
                   (frequent_itemsets1['support'] >= 0.015) ] #Para parejas poner lengh en 2 y support mayor que 0.023, y para trios poner length en3 y support inferior a 0.005


frequent_itemsets1[ (frequent_itemsets1['length'] == 3) &
                   (frequent_itemsets1['support'] >= 0.003344) ] #Para parejas poner lengh en 2 y support mayor que 0.023, y para trios poner length en3 y support inferior a 0.005

# estrellas
te2 = TransactionEncoder()
te_ary2 = te2.fit(estrellas5).transform(estrellas5)
df2 = pd.DataFrame(te_ary2, columns=te2.columns_)
df2
frequent_itemsets2 = apriori(df2, min_support=0.001, use_colnames=True)
frequent_itemsets2['length'] = frequent_itemsets2['itemsets'].apply(lambda x: len(x))

frequent_itemsets2[ (frequent_itemsets2['length'] == 2) &
                   (frequent_itemsets2['support'] >= 0.025) ] #Para parejas poner lengh en 2 y support mayor que 0.023, y para trios poner length en3 y support inferior a 0.005


################################################################
################################################################

