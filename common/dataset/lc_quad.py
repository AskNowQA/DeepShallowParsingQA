import ujson as json


class LC_QuAD:
    def __init__(self, dataset_path):
        with open(dataset_path, 'r') as file_hanlder:
            self.raw_dataset = json.load(file_hanlder)

            self.corpus = [[self.__preprocess(item['corrected_question']),
                            item['corrected_question'],
                            item['annotation']] for item in
                           self.raw_dataset]
            self.validate = all([(len(item[0].split()) == len(item[1])) for item in self.corpus])

    def __preprocess(self, line):
        return line.replace('?', ' ').replace('\'', ' ').replace('-', ' ')
