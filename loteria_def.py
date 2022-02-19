# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 20:17:50 2017

@author: iv
"""
import numpy as np
import pandas as pd
import requests as rq
import urllib
import bs4
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori


####################
### PRIMITIVA ###
####################
url = "http://www.lotoideas.com/primitiva-resultados-historicos-de-todos-los-sorteos/"
print("")

suffix = "=csv"
link_list = []

def getPDFs():
    payload = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:92.0) Gecko/20100101 Firefox/92.0'}
    response = rq.get(url, headers=payload)
    soup = bs4.BeautifulSoup(response.content, features="html.parser")

    for link in soup.find_all('a'):  # Finds all links
        if suffix in str(link):  # If the link ends in .pdf
            link_list.append(link.get('href'))
    print(link_list)


getPDFs()
primitiva1_csv = urllib.request.urlopen(link_list[0]) 
primitiva2_csv = urllib.request.urlopen(link_list[1]) 

primitiva1 = pd.read_csv(primitiva1_csv, index_col=0, parse_dates=False)
primitiva1.index = pd.to_datetime(primitiva1.index)
primitiva1.columns = ['bola_1', 'bola_2', 'bola_3', 'bola_4', 'bola_5', 'bola_6', 'complementario', 'reintegro',
                      'joker']
primitiva2 = pd.read_csv(primitiva2_csv, parse_dates=False)
primitiva2 = primitiva2[primitiva2['FECHA'].str.len() <= 10]
primitiva2 = primitiva2.set_index('FECHA')
primitiva2.index = pd.to_datetime(primitiva2.index)
primitiva2.columns = ['bola_1', 'bola_2', 'bola_3', 'bola_4', 'bola_5', 'bola_6', 'complementario', 'reintegro',
                      'joker']

primitiva = pd.concat([primitiva2, primitiva1], ignore_index=False).sort_index(ascending=False)

primitiva2 = primitiva.iloc[:, 0:7]
primitiva3 = primitiva2.reset_index()
primitiva3 = primitiva3.iloc[:, 1:8]
primitiva4 = primitiva3.astype(str)
primitiva5 = primitiva4.values.tolist()
################################################################
# bolas y complementario
te = TransactionEncoder()
te_ary = te.fit(primitiva5).transform(primitiva5)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.001, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

f = frequent_itemsets[(frequent_itemsets['length'] == 2)].sort_values('support', ascending=False)
reintegro = primitiva['reintegro'].value_counts().sort_values(ascending=False)
#  Para parejas poner lengh en 2 y support mayor que 0.023, y para trios poner length en3 y support inferior a 0.005
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
    payload = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:92.0) Gecko/20100101 Firefox/92.0'}
    response = rq.get(url, headers=payload)
    soup = bs4.BeautifulSoup(response.content, features="html.parser")

    for link in soup.find_all('a'):  # Finds all links
        if suffix in str(link):  # If the link ends in .pdf
            link_list.append(link.get('href'))
    print(link_list)


getPDFs()

euromillones_csv = urllib.request.urlopen(link_list[0])

euromillones1 = pd.read_csv(euromillones_csv, index_col=0, parse_dates=True)
euromillones1.columns = ['bola_1', 'bola_2', 'bola_3', 'bola_4', 'bola_5', 'blanco', 'estrella1', 'estrella2']

euromillones2 = euromillones1.iloc[:, np.r_[0:5]]
euromillones2 = euromillones2.reset_index()

estrellas2 = euromillones1.iloc[:, np.r_[6, 7]]
estrellas2 =  estrellas2.reset_index()

euromillones3 = euromillones2.iloc[:, 1:6]
estrellas3 = estrellas2.iloc[:, 1:3]

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

f = frequent_itemsets1[(frequent_itemsets1['length'] == 2) &
                   (frequent_itemsets1['support'] >= 0.015)]
f.sort_values('support', ascending=False).head(60)

# Para parejas poner lengh en 2 y support mayor que 0.023, y para trios poner length en3 y support inferior a 0.005


f = frequent_itemsets1[(frequent_itemsets1['length'] == 3) &
                   (frequent_itemsets1['support'] >= 0.003344)]
f.sort_values('support', ascending=False).head(60)
# Para parejas poner lengh en 2 y support mayor que 0.023, y para trios poner length en3 y support inferior a 0.005

# estrellas
te2 = TransactionEncoder()
te_ary2 = te2.fit(estrellas5).transform(estrellas5)
df2 = pd.DataFrame(te_ary2, columns=te2.columns_)
frequent_itemsets2 = apriori(df2, min_support=0.001, use_colnames=True)
frequent_itemsets2['length'] = frequent_itemsets2['itemsets'].apply(lambda x: len(x))

f = frequent_itemsets2[(frequent_itemsets2['length'] == 2) &
                       (frequent_itemsets2['support'] >= 0.025)]
# Para parejas poner lengh en 2 y support mayor que 0.023, y para trios poner length en3 y support inferior a 0.005
f.sort_values('support', ascending=False).head(60)
