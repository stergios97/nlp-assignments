#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize

vocab = open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for idx, line in enumerate(vocab):
    #TODO: import part 1 code to build dictionary
        word_index_dict[line.rstrip()] = idx
vocab.close()

len_index = len(word_index_dict)
counts = np.zeros((len_index, len_index)) #TODO: initialize numpy 0s array

#TODO: iterate through file and update counts
with open("brown_100.txt") as f:
    for sentence in f:
        prev_word = False
        for word in sentence.rstrip().rsplit():
            if prev_word:
                counts[word_index_dict[prev_word.lower()], word_index_dict[word.lower()]] += 1
            prev_word = word
            
counts += 0.1
#TODO: normalize counts
probs = normalize(counts, norm='l1', axis=1)

#TODO: writeout bigram probabilities
bigram_list = [('all', 'the'), ('the', 'jury'), ('the', 'campaign'), ('anonymous', 'calls')]

with open('smooth_probs.txt', 'w') as wf:
    for bigram in bigram_list:
        prev_word_index = word_index_dict[bigram[0]]
        word_index = word_index_dict[bigram[1]]
       
        wf.write(str(probs[prev_word_index, word_index]) + '\n')
        
        
# 6. Calculating sentence probabilities 
with open('toy_corpus.txt') as f:
    with open('smoothed_eval.txt', 'w') as wf:
        for line in f:
            sentprob = 1
            sent_len = 0
            prev_word = False
            for word in line.rstrip().rsplit():
                if prev_word:
                    wordprob = probs[word_index_dict[prev_word.lower()], word_index_dict[word.lower()]]
                    sentprob *= wordprob
                    sent_len += 1
                prev_word = word
            perplexity = 1/(pow(sentprob, 1.0/sent_len))
            wf.write(str(perplexity) + '\n')