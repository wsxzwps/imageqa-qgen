from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import argparse
import copy
import pickle as pkl
import logger
import os
import re
import subprocess
import sys
import time

parseFilename = 'train_parsed.txt'
outputFilename = 'noun_blank.txt'

# Logger
log = logger.get()

class TreeNode:
    """Parse tree."""

    def __init__(self, className, text, children, level):
        """Construct a tree.
        """
        self.className = className
        self.text = text
        self.children = children
        self.level = level
        pass

    def __str__(self):
        """To string (with tree structure parentheses)."""
        strlist = []
        for i in range(self.level):
            strlist.append('    ')
        strlist.extend(['(', self.className])
        if len(self.children) > 0:
            strlist.append('\n')
            for child in self.children:
                strlist.append(child.__str__())
            if len(self.text) > 0:
                for i in range(self.level + 1):
                    strlist.append('    ')
            else:
                for i in range(self.level):
                    strlist.append('    ')
        else:
            strlist.append(' ')
        strlist.append(self.text)
        strlist.append(')\n')
        return ''.join(strlist)

    def toSentence(self):
        """Unfold the tree structure into a string."""
        strlist = []
        for child in self.children:
            childSent = child.toSentence()
            if len(childSent) > 0:
                strlist.append(childSent)
        if len(self.text) > 0:
            strlist.append(self.text)
        return ' '.join(strlist)

    def relevel(self, level):
        """Re-assign level."""
        self.level = level
        for child in self.children:
            child.relevel(level + 1)

    def copy(self):
        """Clone a tree."""
        children = []
        for child in self.children:
            children.append(child.copy())
        return TreeNode(self.className, self.text, children, self.level)


class TreeParser:
    """Finite state machine implementation of syntax tree parser."""

    def __init__(self):
        self.state = 0
        self.currentClassStart = 0
        self.currentTextStart = 0
        self.classNameStack = []
        self.childrenStack = [[]]
        self.root = None
        self.rootsList = []
        self.level = 0
        self.stateTable = [self.state0, self.state1, self.state2,
                           self.state3, self.state4, self.state5, self.state6]
        self.raw = None
        self.state = 0

    def parse(self, raw):
        if not self.isAlpha(raw[0]):
            self.raw = raw
            for i in range(len(raw)):
                self.state = self.stateTable[self.state](i)

    @staticmethod
    def isAlpha(c):
        return 65 <= ord(c) <= 90 or 97 <= ord(c) <= 122

    @staticmethod
    def isNumber(c):
        return 48 <= ord(c) <= 57

    @staticmethod
    def exception(raw, i):
        print(raw)
        raise Exception(
            'Unexpected character "%c" (%d) at position %d'
            % (raw[i], ord(raw[i]), i))

    @staticmethod
    def isClassName(s):
        if TreeParser.isAlpha(s) or s in charClassName:
            return True
        else:
            return False

    @staticmethod
    def isText(s):
        if TreeParser.isAlpha(s) or TreeParser.isNumber(s) or s in charText:
            return True
        else:
            return False

    def state0(self, i):
        if self.raw[i] == '(':
            return 1
        else:
            return 0

    def state1(self, i):
        if self.isClassName(self.raw[i]):
            self.currentClassStart = i
            self.level += 1
            self.childrenStack.append([])
            return 2
        else:
            self.exception(self.raw, i)

    def state2(self, i):
        if self.isClassName(self.raw[i]):
            return 2
        else:
            self.classNameStack.append(self.raw[self.currentClassStart:i])
            if self.raw[i] == ' ' and self.raw[i + 1] == '(':
                return 0
            elif self.raw[i] == ' ' and self.isText(self.raw[i + 1]):
                return 4
            elif self.raw[i] == '\n':
                return 3
            else:
                self.exception(self.raw, i)

    def state3(self, i):
        if self.raw[i] == ' ' and self.raw[i + 1] == '(':
            return 0
        elif self.raw[i] == ' ' and self.raw[i + 1] == ' ':
            return 3
        elif self.raw[i] == ' ' and self.isText(self.raw[i + 1]):
            return 4
        else:
            return 3

    def state4(self, i):
        if self.isText(self.raw[i]):
            # global currentTextStart
            self.currentTextStart = i
            return 5
        else:
            self.exception(self.raw, i)

    def state5(self, i):
        if self.isText(self.raw[i]):
            return 5
        elif i == len(self.raw) - 1:
            return 5
        elif self.raw[i] == ')':
            self.wrapup(self.raw[self.currentTextStart:i])
            if self.level == 0:
                return 0
            elif self.raw[i + 1] == ')':
                return 6
            else:
                return 3
        else:
            self.exception(self.raw, i)

    def state6(self, i):
        if self.level == 0:
            return 0
        elif self.raw[i] == ')':
            self.wrapup('')
            return 6
        else:
            return 3

    def wrapup(self, text):
        self.level -= 1
        root = TreeNode(self.classNameStack[-1], text,
                        self.childrenStack[-1][:], self.level)
        del self.childrenStack[-1]
        del self.classNameStack[-1]
        self.childrenStack[-1].append(root)
        if self.level == 0:
            self.rootsList.append(root)



def questionGen(parseFilename, outputFilename=None):
    """Generates questions."""
    startTime = time.time()
    qCount = 0
    numSentences = 0
    parser = TreeParser()
    gen = QuestionGenerator()
    questionAll = []

    def newTree():
        return parser.rootsList[0].copy()

    def addQuestion(sentId, origSent, question, answer, typ):
        questionAll.append((sentId, origSent, question, answer, typ))

    def addItem(qaitem, origSent, typ):
        ques = qaitem[0]
        ans = lookupSynonym(qaitem[1])
        log.info('Question {:d}: {} Answer: {}'.format(
            qCount, ques, ans))
        addQuestion(numSentences, origSent, ques, ans, typ)

    with open(parseFilename) as f:
        for line in f:
            if len(parser.rootsList) > 0:
                origSent = parser.rootsList[0].toSentence()

                # 0 is what-who question type
                for qaitem in gen.askWhoWhat(newTree()):
                    # Ignore too short questions
                    if len(qaitem[0].split(' ')) < 5:
                        # qCount += 1
                        continue
                    qCount += 1
                    addItem(qaitem, origSent, 0)

                # 1 is how-many question type
                for qaitem in gen.askHowMany(newTree()):
                    qCount += 1
                    addItem(qaitem, origSent, 1)

                # 2 is color question type
                for qaitem in gen.askColor(newTree()):
                    qCount += 1
                    addItem(qaitem, origSent, 2)

                # 3 is location question type
                for qaitem in gen.askWhere(newTree()):
                    qCount += 1
                    addItem(qaitem, origSent, 3)

                del(parser.rootsList[0])
                numSentences += 1
            parser.parse(line)

    log.info('Number of sentences: {:d}'.format(numSentences))
    log.info('Time elapsed: {:f} seconds'.format(time.time() - startTime))
    log.info('Number of questions: {:d}'.format(qCount))

    if outputFilename is not None:
        log.info('Writing to output {}'.format(
            os.path.abspath(outputFilename)))
        with open(outputFilename, 'wb') as f:
            pkl.dump(questionAll, f)

def main():
    questionGen(parseFilename, outputFilename)

