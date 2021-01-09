import json


class DataReader:

    @staticmethod
    def get_json(filepath):
        with open(filepath, encoding='utf-8') as json_file:
            return json.load(json_file)

    @staticmethod
    def read_json_to_flat(filepath):
        with open(filepath, encoding='utf-8') as json_file:
            data = json.load(json_file)
            topic_idx = {}
            plain = {}
            for item_name, item_value in data.items():
                concat = ''.join(' '.join(v['text']).rstrip() for v in item_value.values())
                plain[item_name] = concat
                topic_idx[item_name] = []
                for v in item_value.values():
                    if v['topic_num'] is list:
                        topic_idx[item_name].extend(v['topic_num'])
                    else:
                        topic_idx[item_name].append(v['topic_num'])
            return data, plain, topic_idx

