import nltk
import pickle
import os

# nlkt.download('punkt')
# nlkt.download('averaged_perceptron_tagger')

word_dict = {}

def gen(text, parsed_sentence, outputFile):
    for i in range(len(parsed_sentence)):
        if parsed_sentence[i][1] == 'NN':
            new_sentence = text[:]
            if new_sentence[i] in word_dict:
                word_dict[new_sentence[i]] += 1
            else:
                word_dict[new_sentence[i]] = 1
            new_sentence[i] = '[MASK]'
            with open(outputFile, 'a') as f:                
                f.write(' '.join(new_sentence))
                f.write('\n')

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
