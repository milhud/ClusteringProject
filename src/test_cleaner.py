import pandas as pd
import spacy

# utility functions
from utilities.text_cleaner import lemmatize, lowercase, remove_stops, remove_punctuation, tokenize, expand_abbreviation

def clean(text):
    # expanded = expand_abbreviation(text)  need to add
    # lower if needed
    expanded = expand_abbreviation(text)
    doc = tokenize(expanded)              # Get doc object
    lemmatized = lemmatize(doc)       # Lemmatize first (returns string)
    doc2 = tokenize(lemmatized)       # Tokenize the lemmatized string
    no_stops = remove_stops(doc2)     # Remove stops (returns string)  
    doc3 = tokenize(no_stops)         # Tokenize again
    no_punctuation = remove_punctuation(doc3)  # Remove punct (returns string)
    return no_punctuation

# Load first 10 rows
dataframe = pd.read_csv('src/data/questions.csv', sep='|', names=['category', 'question', 'answer'], nrows=10)

# Test your functions
for i in range(1, len(dataframe), 1):
    question = dataframe.iloc[i]['question']
    
    # now clean
    cleaned = clean(question)

    print("Cleaned: ", cleaned)