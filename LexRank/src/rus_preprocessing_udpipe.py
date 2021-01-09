import wget
from ufal.udpipe import Model, Pipeline
import os
import sys
import re


class UdpipeProcessor:

    def __init__(self):
        udpipe_model_url = 'https://rusvectores.org/static/models/udpipe_syntagrus.model'
        modelfile = 'udpipe_syntagrus.model'

        if not os.path.isfile(modelfile):
            print('UDPipe model not found. Downloading...', file=sys.stderr)
            wget.download(udpipe_model_url)
            print('\nLoading the model...', file=sys.stderr)
        self.model = Model.load(modelfile)

    @staticmethod
    def clean_token(token, misc):
        """
        :param token:  токен (строка)
        :param misc:  содержимое поля "MISC" в CONLLU (строка)
        :return: очищенный токен (строка)
        """
        out_token = token.strip().replace(' ', '')
        if token == 'Файл' and 'SpaceAfter=No' in misc:
            return None
        return out_token

    @staticmethod
    def clean_lemma(lemma, pos):
        """
        :param lemma: лемма (строка)
        :param pos: часть речи (строка)
        :return: очищенная лемма (строка)
        """
        out_lemma = lemma.strip().replace(' ', '').replace('_', '').lower()
        if '|' in out_lemma or out_lemma.endswith('.jpg') or out_lemma.endswith('.png'):
            return None
        if pos != 'PUNCT':
            if out_lemma.startswith('«') or out_lemma.startswith('»'):
                out_lemma = ''.join(out_lemma[1:])
            if out_lemma.endswith('«') or out_lemma.endswith('»'):
                out_lemma = ''.join(out_lemma[:-1])
            if out_lemma.endswith('!') or out_lemma.endswith('?') or out_lemma.endswith(',') \
                    or out_lemma.endswith('.'):
                out_lemma = ''.join(out_lemma[:-1])
        return out_lemma

    @staticmethod
    def num_replace(word):
        newtoken = 'x' * len(word)
        return newtoken

    def process(self, pipeline, text='Строка', keep_pos=True, keep_punct=False):
        entities = {'PROPN'}
        named = False
        memory = []
        mem_case = None
        mem_number = None
        tagged_propn = []

        processed = pipeline.process(text)

        content = [l for l in processed.split('\n') if not l.startswith('#')]

        tagged = [w.split('\t') for w in content if w]

        for t in tagged:
            if len(t) != 10:
                continue
            (word_id, token, lemma, pos, xpos, feats, head, deprel, deps, misc) = t
            token = self.clean_token(token, misc)
            lemma = self.clean_lemma(lemma, pos)
            if not lemma or not token:
                continue
            if pos in entities:
                if '|' not in feats:
                    tagged_propn.append('%s_%s' % (lemma, pos))
                    continue
                morph = {el.split('=')[0]: el.split('=')[1] for el in feats.split('|')}
                if 'Case' not in morph or 'Number' not in morph:
                    tagged_propn.append('%s_%s' % (lemma, pos))
                    continue
                if not named:
                    named = True
                    mem_case = morph['Case']
                    mem_number = morph['Number']
                if morph['Case'] == mem_case and morph['Number'] == mem_number:
                    memory.append(lemma)
                    if 'SpacesAfter=\\n' in misc or 'SpacesAfter=\s\\n' in misc:
                        named = False
                        past_lemma = '::'.join(memory)
                        memory = []
                        tagged_propn.append(past_lemma + '_PROPN ')
                else:
                    named = False
                    past_lemma = '::'.join(memory)
                    memory = []
                    tagged_propn.append(past_lemma + '_PROPN ')
                    tagged_propn.append('%s_%s' % (lemma, pos))
            else:
                if not named:
                    if pos == 'NUM' and token.isdigit():  # Заменяем числа на xxxxx той же длины
                        lemma = self.num_replace(token)
                    tagged_propn.append('%s_%s' % (lemma, pos))
                else:
                    named = False
                    past_lemma = '::'.join(memory)
                    memory = []
                    tagged_propn.append(past_lemma + '_PROPN ')
                    tagged_propn.append('%s_%s' % (lemma, pos))

        if not keep_punct:
            tagged_propn = [word for word in tagged_propn if word.split('_')[1] != 'PUNCT']
        if not keep_pos:
            tagged_propn = [word.split('_')[0] for word in tagged_propn]
        return tagged_propn

    def tag_ud(self, text='Текст нужно передать функции в виде строки!'):
        process_pipeline = Pipeline(self.model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        res = []
        for line in text.split('.'):
            output = self.process(process_pipeline, text=line)
            res.append(output)
        return res



