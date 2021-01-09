from biasedLexRank import BiasedLexRank
from data_reader import DataReader
from text_preprocessing import TextPreprocessor
from rus_preprocessing_udpipe import UdpipeProcessor
import json
import metrics as m
import numpy as np


def start_work(d_value):
    reader = DataReader()
    data, plain_texts, topic_idx = reader.read_json_to_flat('texts/min_person_all.json')
    topics = reader.get_json('texts/topics_artm.json')
    udpipe_proc = UdpipeProcessor()

    for k, v in data.items():
        for head, value in data[k].items():
            s = ' '.join(data[k][head]['text']).replace('\n', '').strip().strip('.').split('.')
            data[k][head]['len'] = len(s)

    processed = 0
    failed_topic = 0
    s_count = 3
    d = d_value
    summaries = {}

    summarizer = BiasedLexRank()

    for key, text in plain_texts.items():
        is_ok = False
        sentences = udpipe_proc.tag_ud(text.strip('.'))
        a = data[key]
        if any([data[key][headline]['len'] <= s_count for headline in data[key].keys()]):  # or len(topic_idx[key]) < 2
            continue

        for t in topic_idx[key]:
            topic = t
            if type(t) is list:
                topic = t[0]  # TODO: переделать на списки все
            if topic > 49:
                failed_topic += 1
                continue
            topic_bias = ' '.join(topics[f'topic_{topic}'])
            topic_bias = udpipe_proc.tag_ud(topic_bias)

            idx = summarizer(sentences, topic_bias, d=d, sentences_count=s_count)
            s = text.replace('\n', '').strip('.').split('.')
            is_ok = True
            summaries[f'{key}_{topic}'] = {'bias': topic_bias, 'summary_list': [s[i] for i in idx]}

        processed += is_ok
        print(processed)

    print(processed)

    with open(f"results/summaries_s{s_count}_d{d}.json", "w", encoding='utf-8') as outfile:
        json.dump(summaries, outfile, ensure_ascii=False)
