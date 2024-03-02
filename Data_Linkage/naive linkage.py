import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
import textdistance as td
from nltk.stem import WordNetLemmatizer 

# pre-process a string, of which the functionality includes: 1. deal with missing value; 2. convert to lower case; 3. remove not only punctuations attaching to word but those inside word; 4. remove stopwords 
def process_string(text):
    test_list = ['missing']
    if pd.notna(text):
        text = text.lower()
        test_list = nltk.word_tokenize(text)
        test_list = [word for word in test_list if word not in stopword]
        test_list = [word for word in test_list if word not in string.punctuation]
        for i in range(len(test_list)):
            test_list[i] = ''.join([char for char in test_list[i] if char not in string.punctuation])          
    return test_list

# This function try to pre-process the dataframe for the record linkage
def pre_process(df):
    # remove noise
    df = df.replace('#NAME?', np.NaN)
    # add a new column to record serial number
    df['number'] = ''
    for index, row in df.iterrows():
        unit = 1;
        # pre-process name column
        text = row['name'].lower()
        text_list = nltk.word_tokenize(text)
        if len(text_list) > 2:
            if text_list[-2] == '-':
                number = ''.join([char for char in text_list[-1] if char not in string.punctuation])
                df.at[index, 'number'] = number
        
        df.at[index, 'name'] = process_string(text)
        
        # pre-process description column
        if pd.notna(row['description']):
            text = row['description'].lower()
            # try to strip order number (for some records), for buy.csv
            search = re.search(r'^\d+\sx', text)
            if search:
                unit = int(re.findall(r'\d+', search.group(0))[0])
                text = re.sub(r'^\d+\sx', '', text)
            
            # try to strip repeated name content re-appear in description, for abt.csv
            text = re.sub(row['name'], '', text)
            
            df.at[index, 'description'] = process_string(text)
        else:
            df.at[index, 'description'] = []

        # pre-process price column
        if pd.notna(row['price']):
            price = float(re.sub(r'[^\d.]', '', row['price']))
            # adjust to unit price according to the order numbe extracted from description and total price, for buy.csv
            df.at[index, 'price'] = price / unit
    return df



abt = pd.read_csv('abt_small.csv', encoding = "ISO-8859-1")
buy = pd.read_csv('buy_small.csv', encoding = "ISO-8859-1")

stopword = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
threshold = 0.435

idAbt = []
idBuy = []

abt = pre_process(abt)
buy = pre_process(buy)

# also process manufacturer column for buy
for index, row in buy.iterrows():
    buy.at[index, 'manufacturer'] = process_string(row['manufacturer']);

# naive data linkage implementation
for index1, row1 in abt.iterrows():
    name_score = 0
    desc_score = 0
    max_score = 0
    sum_score = 0
    for index2, row2 in buy.iterrows():
        # compare name similarity
        name_score = td.cosine(row1['name'], row2['name'])
        
        # compare brand
        if abt.loc[index1, 'name'][0] != buy.loc[index2, 'name'][0] and abt.loc[index1, 'name'][0] != buy.loc[index2, 'manufacturer'][0]:
            name_score = -10;
        
        # compare description similarity
        if buy.loc[index2, 'description']:
            desc_score = td.cosine(abt.loc[index1, 'description'], buy.loc[index2, 'description'])
            
        # detect whether difference in price are relatively large
        if pd.notna(abt.loc[index1, 'price']) and pd.notna(buy.loc[index2, 'price']):
            # calculate relative price difference
            price_diff = row1['price'] - row2['price']
            percentage_diff = abs(price_diff) / min(row1['price'], row2['price'])
            # two product with completely different price is unusual to be the same one 
            if percentage_diff > 0.4:
                name_score = -10
                
        # compare serial number
        if abt.loc[index1, 'number'] and buy.loc[index2, 'number']:
            if abt.loc[index1, 'number'] == buy.loc[index2, 'number']:
                name_score = 10
            
        # weighted score formula
        sum_score = 0.8 * name_score + 0.1 * desc_score
        if sum_score > max_score:
            max_score = sum_score
            idA = row1['idABT']
            idB = row2['idBuy']
    
    if (max_score > threshold):
        idAbt.append(idA)
        idBuy.append(idB)

# output results
task_1 = pd.DataFrame({'idAbt': idAbt, 'idBuy': idBuy})
task_1.to_csv('task1a.csv', index = False)

