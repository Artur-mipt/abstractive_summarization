import re
import gensim
import nlp
import numpy as np
from scipy import spatial
from nltk.corpus import stopwords
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import spacy
from math import log


def process_str(s):
    s = s.lower()
    s = re.sub("[^0-9a-zA-Z ]", " ", s)
    return s


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        
        
def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, trigram_mod, bigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def process_texts(texts, bigram=False, lemmas=False):
    words = list(sent_to_words(texts))
    words = remove_stopwords(words)

    if bigram:
        bigram = gensim.models.Phrases(words, min_count=5, threshold=100)
        trigram = gensim.models.Phrases(bigram[words], threshold=100)  
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        words = make_bigrams(words, bigram_mod)
    
    if lemmas:
        words = lemmatization(words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    return words


def get_topic_vector(lda_model, text, num_topics=10):
    doc_topic_vector = lda_model.get_document_topics(text)
    res = np.zeros(num_topics)
    for pair in doc_topic_vector:
        res[pair[0]] = pair[1]
    return list(res)


def get_ohe_topic_vector(lda_model, text):
    doc_topic_vector = lda_model.get_document_topics(text)
    doc_topic_vector = [a[1] for a in doc_topic_vector]
    max_ind = np.argmax(doc_topic_vector)
    zeros = np.zeros(len(doc_topic_vector))
    zeros[max_ind] = 1
    return zeros


def cosine(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)


def cosine_dist(v1, v2):
    return spatial.distance.cosine(v1, v2)


def euclidean_dist(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))


def plot_distances(articles_topics, highlights_topics):
    cosines = []
    euclidean_dists = []
    for article_topic, highlight_topic in zip(articles_topics, highlights_topics):
        try:
            cosines.append(cosine(article_topic, highlight_topic))
            euclidean_dists.append(euclidean_dist(article_topic, highlight_topic))
        except:
            pass
    
    cosines = np.array(cosines)
    euclidean_dists = np.array(euclidean_dists)
    
    cos_mean = round(np.mean(cosines), 3)
    cos_std = round(np.std(cosines), 3)
    eucl_mean = round(np.mean(euclidean_dists), 3)
    eucl_std = round(np.std(euclidean_dists), 3)
    
    print(f'cosines: mean {cos_mean}, std {cos_std}')
    print(f'euclidean distances: mean {eucl_mean}, std {eucl_std}')
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('cosines')
    plt.hist(cosines, bins=20)

    plt.subplot(122)
    plt.title('euclidean distances')
    plt.hist(euclidean_dists, bins=20)
    plt.show()
    

def beam_search_decoder(data, k):
    sequences = [[list(), 0.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score - log(row[j])]
                all_candidates.append(candidate)
            # order all candidates by score
            all_candidates = sorted(all_candidates, key=lambda tup:tup[1])[:k]
        # select k best
        sequences = all_candidates[:k]
        
    return sequences


def tensor_to_text(t, sp, beam_search=False, beam_size=3):
    if len(t.size()) == 3 and beam_search:
        t = t.permute(1, 0, 2)  # (batch_size, seq_len, vocab_size)
        t = F.softmax(t, dim=2).detach().cpu().numpy()
        tokens = []
        for sentence in t:
            beam_search_res = beam_search_decoder(sentence, k=beam_size)
            probs = np.exp(np.array([-a[1] for a in beam_search_res]))
            probs /= np.sum(probs)
            sentences = [a[0] for a in beam_search_res]
            # res = np.random.choice(len(sentences), size=1, p=probs)[0]
            res = np.random.choice(len(sentences), size=1)[0]
            tokens.append(sentences[res])
        tokens = list(map(lambda x: [int(c) for c in x], tokens))
        return sp.decode(tokens)
    elif len(t.size()) == 3:  # tensor of scores
        t = t.permute(1, 0, 2)
        t = F.softmax(t, dim=2)
        tokens = torch.argmax(t, dim=2).detach().cpu().numpy().tolist()
        tokens = list(map(lambda x: [int(c) for c in x], tokens))
        return sp.decode(tokens)
    elif len(t.size()) == 2:  # tensor of tokens
        tokens = t.permute(1, 0).detach().cpu().numpy().tolist()
        tokens = list(map(lambda x: [int(c) for c in x], tokens))
        return sp.decode(tokens)