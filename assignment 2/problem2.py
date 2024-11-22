#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np

vocab = open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    #TODO: import part 1 code to build dictionary
    word_index_dict[line.rstrip()] = i

#TODO: initialize counts to a zero vector
counts = np.zeros(len(word_index_dict))

#TODO: iterate through file and update counts
with open("brown_100.txt") as f:
    for sentence in f:
        for word in sentence.rstrip().rsplit():
            counts[word_index_dict[word.lower()]] += 1

#TODO: normalize and writeout counts. 
probs = counts / np.sum(counts)
with open('unigram_probs.txt', 'w') as wf:
    wf.write(str(probs))

# 6. Calculating sentence probabilities 
with open('toy_corpus.txt') as f:
    with open('unigram_eval.txt', 'w') as wf:
        for line in f:
            sentprob = 1
            sent_len = 0
            for word in line.rstrip().rsplit():
                wordprob = probs[word_index_dict[word.lower()]]
                sentprob *= wordprob
                sent_len += 1
            perplexity = 1/(pow(sentprob, 1.0/sent_len))
            wf.write(str(perplexity) + '\n')
        