import pandas as pd

def parse_questions(raw_text):
    lines = raw_text.strip().split("\n")
    questions_data = []
    prevQuestion = False

    for line in lines:

        # skip empty lines
        if line.strip() == "":
            continue
        elif prevQuestion:
            prevQuestion = False
            continue

        # this is the category; doesn't end with ? or .
        if not line[0].isspace():
            current_category = line.strip()
            print("New Category: ", current_category)

        # this is the question
        elif not prevQuestion:
            question = line.strip()
            questions_data.append([current_category, question])
            prevQuestion = True
        
        else:
            continue

    return questions_data

# open and process the file
file = open("../data/raw_text.txt", "r", encoding="utf-8") # encoding: fix due to weird double quotes?
text = file.read()

# parse questions, convert to pandas dataframe
questions = parse_questions(text)
dataframe = pd.DataFrame(questions)

# save as CSV, close file
dataframe.to_csv("../data/questions.csv", index=False, sep="|") # some questions have commas in them so need another separator
file.close()

# confirmation message
print("Data saved successfully.")
