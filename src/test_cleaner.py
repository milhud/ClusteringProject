import pandas as pd
import spacy

# utility functions
from utilities.text_cleaner import lemmatize, lowercase, remove_stops, remove_punctuation, tokenize, expand_abbreviation

# somewhat helpful reference: https://blog.gopenai.com/from-messy-text-to-model-ready-data-a-guide-to-nlp-preprocessing-51323efc3876
def clean(text):
    # expanded = expand_abbreviation(text)  need to add
    # lower if needed
    expanded = expand_abbreviation(text)
    doc = tokenize(expanded)              
    lemmatized = lemmatize(doc) # first lemmatize
    doc2 = tokenize(lemmatized)       
    no_stops = remove_stops(doc2)
    doc3 = tokenize(no_stops) 
    no_punctuation = remove_punctuation(doc3)  # remove the punctuation 
    return no_punctuation

# first ten
dataframe = pd.read_csv('src/data/questions.csv', sep='|', names=['category', 'question', 'answer'], nrows=10)

# simple clean test
print("First ten cleaned:")
for i in range(1, len(dataframe), 1):
    question = dataframe.iloc[i]['question']
    
    # now clean
    cleaned = clean(question)

    print("Cleaned: ", cleaned)