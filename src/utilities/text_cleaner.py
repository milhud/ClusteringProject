from data.abbreviations import ABBREVIATIONS
import spacy

# used for tagging and lemmatizing, as well as identifying stop words; small module
tokenize = spacy.load("en_core_web_sm")

def tokenize_text(text):
    tokens = tokenize(text)
    return tokens

# e.g. changing, changes -> change; for uniform meaning; also converts to lowercase
def lemmatize(tokens):
    lemmatized_tokens = []
    
    for token in tokens:
        lemmatized_tokens.append(token.lemma_.lower())
    
    return " ".join(lemmatized_tokens)

def lowercase(line):
    return line.lower()

def remove_stops(tokens):
    cleaned_tokens = []
    for token in tokens:
        if not token.is_stop:
            cleaned_tokens.append(token.text)
    return " ".join(cleaned_tokens)

def expand_abbreviation(line):
    words = line.split()  # into the words
    ret = []
    
    for word in words:
        # remove punctuation; the remove punctuation function expects a job object so let's just do it manually
        clean_word = word.replace('?', '').replace('.', '').replace(',', '').replace('!', '')
        
        # in abbreviations; need to be uppercase
        if clean_word.upper() in ABBREVIATIONS:
            expanded = ABBREVIATIONS[clean_word.upper()] # because the acronyms are in upper case
            ret.append(expanded)
        else:
            ret.append(word)
    
    return ' '.join(ret)
            

def remove_punctuation(tokens):
    cleaned_tokens = []
    for token in tokens:
        if not token.is_punct:
            cleaned_tokens.append(token.text)
    return " ".join(cleaned_tokens)

#def expand_abbreviation(line):

