# import json
# with open('data/train.json', 'r') as f:
#     data = json.load(f)

# sentences = []
# for key in data:
#     sentences.extend(data[key]['sentences'])

# with open('train.txt', 'w', encoding='utf-8') as f:
#     for sentence in sentences:
#         if len(sentence.split()) > 30:
#             continue
#         f.write(sentence.strip())
#         f.write('\n')

import pickle

with open('question.pkl','rb') as f:
    q = pickle.load(f)

with open('questions.txt', 'w') as f:
    for question in q:
        f.write(str(question[0])+'\t'+question[1]+'\t'+question[2]+'\t'+question[3]+'\n')