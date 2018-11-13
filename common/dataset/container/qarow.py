from common.dataset.container.sparql import SPARQL
from nltk.corpus import stopwords


class QARow:
    stop_words = set(stopwords.words('english'))

    def __init__(self, question, annotation, raw_sparql, remove_entity_mention, remove_stop_words):
        self.question = question
        self.sparql = SPARQL(raw_sparql)
        self.normalized_question = self.__preprocess(self.question, remove_entity_mention, remove_stop_words)
        self.annotation = annotation

    def validate(self):
        return len(self.normalized_question) == len(self.annotation)

    def __preprocess(self, line, remove_entity_mention, remove_stop_words):
        line = line.lower().replace('?', ' ').replace('\'', ' ').replace('-', ' ').replace(',', ' ')
        entity_labels = ''.join([entity.label for entity in self.sparql.entities])
        output = []
        for word in line.split():
            if remove_entity_mention and (word in entity_labels and word not in QARow.stop_words):
                if len(output) == 0 or output[-1] != '<ent>':
                    output.append('<ent>')
            elif remove_stop_words and word in QARow.stop_words:
                pass
            else:
                output.append(word)
        return output
