from functools import reduce
import numpy as np


class BiasedLexRank:

    def __call__(self, sentences, topic_description, d, sentences_count):
        self.topic_description = topic_description
        self.d = d
        self.sentences = [s.split() for s in sentences]
        self.text_length = reduce(lambda count, s: count + len(s), self.sentences, 0)
        self.baseline_vector = self.baseline_ranking()
        self.matrix = self.build_similarity_matrix()
        lex_rank_scores = self.do_lex_rank(10e-3)
        return self.get_sentences_ids(lex_rank_scores, sentences_count)

    def baseline_ranking(self):
        bias_nodes = []
        for sentence in self.sentences:
            bias = self._get_gen_sentence_probability(self.topic_description, sentence)
            bias_nodes.append(bias)
        return np.array(bias_nodes)

    def _get_gen_sentence_probability(self, sentence_u, sentence_v):
        p = 1
        for word in sentence_u:
            p *= self._get_gen_word_probability(word, sentence_v)
        return p ** (1 / len(sentence_u))

    @staticmethod
    def _get_gen_word_probability(word, sentence, smooth_coef=0.5):
        # TODO: smoothing
        p_in_sentence = sentence.count(word) / len(sentence)
        return 1e-10 if p_in_sentence == 0 else p_in_sentence

    def build_similarity_matrix(self):
        sentences_count = len(self.sentences)
        matrix = np.zeros((sentences_count, sentences_count))
        for i in range(sentences_count):
            for j in range(i, sentences_count):
                if i != j:
                    val = self._get_gen_sentence_probability(self.sentences[i], self.sentences[j])
                    matrix[i][j] = val
                    matrix[j][i] = val
        return self.d * self.baseline_vector + (1 - self.d) * matrix

    def do_lex_rank(self, epsilon):
        sentences_count = len(self.sentences)
        probabilities = np.ones(sentences_count) / sentences_count
        diff = 1
        while diff > epsilon:
            tmp = np.dot(self.matrix.T, probabilities)
            diff = np.linalg.norm(np.subtract(tmp, probabilities))
            probabilities = tmp
        return probabilities

    @staticmethod
    def get_sentences_ids(lex_rank_scores, sentences_count):
        sorted_ids = [i[0] for i in sorted(enumerate(lex_rank_scores), key=lambda x:x[1], reverse=True)]
        return sorted_ids[:sentences_count]
