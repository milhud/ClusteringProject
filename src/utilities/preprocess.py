import pandas as pd

def parse_questions(raw_text):
    lines = raw_text.strip().split("\n")
    questions_data = []
    prevQuestion = False
    
    i = 0
    while i < len(lines):
        line = lines[i]

        # skip empty lines
        if line.strip() == "":
            i += 1
            continue

        # this is the category; doesn't end with ? or .
        if not line[0].isspace():
            current_category = line.strip()
            print("New Category: ", current_category)
            i += 1
            continue

        # this is the question
        else:
            question = line.strip()
            answer = lines[i+1].strip()
            questions_data.append([current_category, question, answer])
            i += 2
            continue

    return questions_data

# open and process the file
file = open("data/raw_text.txt", "r", encoding="utf-8") # encoding: fix due to weird double quotes?
text = file.read()

# parse questions, convert to pandas dataframe
questions = parse_questions(text)
dataframe = pd.DataFrame(questions)

# save as CSV, close file
dataframe.to_csv("data/questions.csv", index=False, sep="|") # some questions have commas in them so need another separator
file.close()

# confirmation message
print("Data saved successfully.")
