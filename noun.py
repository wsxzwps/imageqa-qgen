import nltk

def gen(text, parsed_sentence, outputFile):
    for i in range(len(parsed_sentence)):
        if parsed_sentence[i][1] == 'NN':
            new_sentence = text[:]
            new_sentence[i] = '[MASK]'
            with open(outputFile, 'w') as f:
                f.write(new_sentence)
                f.write('\n')

def questionGen(inputFile, outputFile):
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

if __name__ == '__main__':
    main()
