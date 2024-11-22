import nltk
import math
from collections import Counter

nltk.download('brown')
corpus = nltk.corpus.brown.words()

# calculate word frequencies
word_freq = Counter(corpus)

# select only words that appear more than 10 times
vocab = set([w for w in corpus if word_freq[w] >= 10])

# calculate pair frequencies
pair_freq = Counter([(corpus[i], corpus[i+1]) for i in range(len(corpus)-1) if corpus[i] in vocab and corpus[i+1] in vocab])

# calculate probabilities
pair_prob = {pair: freq*len(vocab) for pair, freq in pair_freq.items()}
word_prob = {w: freq for w, freq in word_freq.items() if w in vocab}

# calculate PMI
pmi = {}
for pair in pair_prob:
    w1, w2 = pair
    pmi_value = math.log2(pair_prob[pair] / (word_prob[w1] * word_prob[w2]))
    pmi[pair] = pmi_value

# get 20 highest and lowest PMI pairs
highest_pmi_pairs = sorted(pmi.items(), key=lambda x: x[1], reverse=True)[:20]
lowest_pmi_pairs = sorted(pmi.items(), key=lambda x: x[1])[:20]

print("20 highest PMI pairs:")
for pair, score in highest_pmi_pairs:
    print(pair[0], pair[1], score)
    
print("20 lowest PMI pairs:")
for pair, score in lowest_pmi_pairs:
    print(pair[0], pair[1], score)