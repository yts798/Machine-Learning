import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords

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

# pre-process the dataframe for blocking    
def pre_process(df):
    # add a column to store product brand as blocks
    df = df.replace('#NAME?', np.NaN)
    df['brand'] = ''
    for index, row in df.iterrows():
        # pre-process 'name' column for both abt and buy file
        df.at[index, 'name'] = process_string(row['name'])
        
        # only pre-process 'manufacturer' and 'description' for buy file
        if 'manufacturer' in df:
            # pre-process 'description' column
            df.at[index, 'description'] = process_string(row['description'])
                
            # pre-process 'manufacturer' column
            df.at[index, 'manufacturer'] = process_string(row['manufacturer'])
    return df

abt = pd.read_csv('abt.csv', encoding = "ISO-8859-1")
buy = pd.read_csv('buy.csv', encoding = "ISO-8859-1")

stopword = stopwords.words('english')

abt = pre_process(abt)
buy = pre_process(buy)

# store block keys
block = []

# extract all first pre-processed word in name as block keys
# assign each columns to block
for index, row in abt.iterrows():
    word = row['name'][0]
    abt.at[index, 'brand'] = word
    if word not in block:
        block.append(word)
        
# match all records from buy to block keys
# try to match first word from buy information, in the priority of 'name', 'manufacturer' and 'description'
for index, row in buy.iterrows():
    if row['name'][0] in block:
        buy.at[index, 'brand'] = row['name'][0]
    elif row['manufacturer'][0] in block:
        buy.at[index, 'brand'] = row['manufacturer'][0]
    elif row['description'][0] in block:
        buy.at[index, 'brand'] = row['description'][0]
    # fail to match
    else:
        buy.at[index, 'brand'] = 'nobrand'

# only keep records that find a match
buy = buy[buy.brand != 'nobrand'];

task1b_abt = pd.DataFrame({'block_key': abt['brand'], 'product_id': abt['idABT']})
task1b_abt.to_csv('abt_blocks.csv', index = False)

task1b_buy = pd.DataFrame({'block_key': buy['brand'], 'product_id': buy['idBuy']})
task1b_buy.to_csv('buy_blocks.csv', index = False)




