import pandas as pd
print("imported pandas")
from bertopic import BERTopic
print("imported")

# load dataframe
dataframe = pd.read_csv("../data/questions.csv", sep="|", names=["category", "question"])

