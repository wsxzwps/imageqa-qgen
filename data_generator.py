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

import json
import pickle
with open('data/train.json', 'r') as f:
    data = json.load(f)

id_dictionary = []

for key in data:
    for i in range(len(data[key]['sentences'])):
        id_dictionary.append((key, data[key]['timestamps'][i]))

with open('questions_subject.pkl', 'rb') as f:
    q = pickle.load(f)

with open('q.txt','w') as fq:
    with open('a.txt', 'w') as fa:
        for question in q:
            question_time_pair = id_dictionary[question[0]]
            fq.write(question_time_pair[0]+'\t'+str(question_time_pair[1])+'\n'+question[2]+'\n')
            fq.write('\n')
            fa.write(question[3]) 
            fa.write('\n')

# import pickle
# import json

# with open('questions_Who.pkl','rb') as f:
#     q = pickle.load(f)

# word_dict = {}

# with open('questions_Who.txt', 'w') as f:
#     for question in q:
#         f.write(str(question[0])+'\t'+question[1]+'\t'+question[2]+'\t'+question[3]+'\n')
#         sentence = question[2].split()
#         if len(sentence) < 3:
#             continue
#         if sentence[0] not in word_dict:
#             word_dict[sentence[0]] = {}
#         if sentence[1] not in word_dict[sentence[0]]:
#             word_dict[sentence[0]][sentence[1]] = {}
#         if sentence[2] not in word_dict[sentence[0]][sentence[1]]:
#             word_dict[sentence[0]][sentence[1]][sentence[2]] = 1
#         else:
#             word_dict[sentence[0]][sentence[1]][sentence[2]] += 1


# from collections import Counter, defaultdict
# import json
# import pickle
# from typing import Any, Dict, Iterator, List

# # from conllu import parse

# MAX_INDEX = 2  # We suppose all sentences have length at least MAX_INDEX + 1.


# def create_empty_counter(i: int = 0) -> Dict[str, Any]:
#     empty_counter = {'counter': Counter()}
#     if i < MAX_INDEX:
#         empty_counter['sub_counters'] = defaultdict(lambda: create_empty_counter(i=i + 1))
#     return empty_counter


# def count(sentence: Iterator[str], counter_dict: Dict[str, Any]) -> None:
#     token = next(sentence)

#     counter_dict['counter'][token] += 1

#     if 'sub_counters' in counter_dict:
#         count(sentence, counter_dict['sub_counters'][token])


# def to_plot_format(counter_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
#     return_list = []

#     counter_dict_counter = counter_dict['counter']
#     for token, size in counter_dict_counter.most_common():
#         token_dict = {'name': token}
#         sub_counters = counter_dict.get('sub_counters')
#         if sub_counters:
#             token_dict['children'] = to_plot_format(sub_counters[token])
#         else:
#             token_dict['size'] = size
#         return_list.append(token_dict)

#     return return_list


# def normalize(token: Dict[str, Any]) -> str:
#     token_form = token.lower()

#     if token_form == '\'s':  # All the 's in the graph are actually "is".
#         token_form = 'is'

#     return token_form

# def main():

#     with open('questions_Who.pkl','rb') as f:
#         q = pickle.load(f)
    
#     sentences = []

#     for question in q:
#         sentences.append(question[2].split())

#     counter_dict = create_empty_counter()
#     for sentence in sentences:
#         count((normalize(token) for token in sentence), counter_dict)

#     count_list = to_plot_format(counter_dict)

#     with open('sunburst_plot_data.json', 'w') as file:
#         json.dump({'name': '', 'children': count_list}, file, indent=2)


# if __name__ == '__main__':
#     main()