import nltk
import pickle
import os
import random

# nlkt.download('punkt')
# nlkt.download('averaged_perceptron_tagger')

word_dict = {}
articles = ['a', 'an']
possessive_pronouns = ['his', 'her', 'their', 'its', 'your', 'my']

def gen(text, parsed_sentence, outputFile):
    position = []
    for i in range(len(parsed_sentence)):
        if parsed_sentence[i][1] == 'NN':
            position.append(i)
    if len(position): 
        idx = random.sample(position, 1)[0]
        new_sentence = text[:]
        correct_word = new_sentence[idx]
        if correct_word in word_dict:
            word_dict[correct_word] += 1
        else:
            word_dict[correct_word] = 1
        new_sentence[idx] = '[MASK]'
        if idx > 0:
            if text[idx - 1].lower() in articles:
                if idx == 1:
                    text[idx - 1] = 'A/An'
                else:
                    text[idx - 1] = 'a/an'
            if text[idx - 1].lower() in possessive_pronouns:
                if idx == 1:
                    text[idx - 1] = '/'.join([w.capitalize() for w in possessive_pronouns])
                else:
                    text[idx - 1] = '/'.join(possessive_pronouns)

        with open(outputFile, 'a') as f:                
            f.write(' '.join(new_sentence))
            f.write('\t')
            f.write(correct_word)
            f.write('\n')
    else:
        return
    

def questionGen(inputFile, outputFile):
    if os.path.exists(outputFile):
        os.remove(outputFile)
    with open(inputFile, 'r') as f:
        line = f.readline()
        while line:
            text = nltk.word_tokenize(line.strip())
            parsed_sentence = nltk.pos_tag(text)
            gen(text, parsed_sentence, outputFile)
            line = f.readline()

def main():
    inputFile = 'train.txt'
    outputFile = 'noun_blank.txt'
    questionGen(inputFile, outputFile)
    with open('nouns.pkl', 'wb') as f:
        pickle.dump(word_dict, f)

if __name__ == '__main__':
    main()
