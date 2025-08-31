import pandas as pd




def parse_questions(raw_text):
    lines = raw_text.strip().split("\n")
    questions_data = []

    for line in lines:

        # skip empty lines
        if line.strip() == "":
            continue

        # this is the category
        if not line.startswith("\t") and not line.strip().endswith("?"):
            current_category = line.strip()

        # this is the question
        elif line.strip().endswith("?"):
            question = line.strip()

            questions_data.append([current_category, question])

    return questions_data

# open and process the file
file = open("../data/raw_text.txt", "r", encoding="utf-8")
text = file.read()

questions = parse_questions(text)

dataframe = pd.DataFrame(questions)

dataframe.to_csv("../data/questions.csv", index=False)

file.close()

print("Data saved successfully.")
