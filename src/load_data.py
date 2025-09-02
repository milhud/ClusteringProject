import pandas as pd
from test_cleaner import clean

def loadPreprocess():
    
    dataframe = pd.read_csv("./src/data/questions.csv", sep="|", names = ["category", "question", "answer"])

    questions = dataframe["question"].tolist()
    answers = dataframe["answer"].tolist()
    category = dataframe["category"].tolist()

    # now clean questions, answers
    cleaned_questions = []
    for question in questions:
        cleaned_questions.append(clean(question))
    
    cleaned_answers = []
    for answer in answers:

