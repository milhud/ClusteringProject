import pandas as pd
from test_cleaner import clean

def loadPreprocess():
    
    dataframe = pd.read_csv("./src/data/questions.csv", sep="|", names = ["category", "question", "answer"])

    questions = dataframe["question"].tolist()
    answers = dataframe["answer"].tolist()
    category = dataframe["category"].tolist()

    # now clean questions, answers and combine
    cleaned_questions = []
    for question in questions:
        cleaned_questions.append(clean(question))
    
    cleaned_answers = []
    for answer in answers:
        cleaned_answers.append(clean(answer))

    combined = []
    for i in range(len(cleaned_questions)):
        question = cleaned_questions[i] # should be same size 
        answer = cleaned_answers[i]
        combined.append(f"{question} {answer}")
    
    return combined, category, cleaned_questions # latter two for ground truth and clustering on questions alone

