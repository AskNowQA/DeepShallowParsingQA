from common.dataset.container.sparql import SPARQL
from nltk.corpus import stopwords


class QARow:
    stop_words = set(stopwords.words('english') + ['whose']) - set(['where'])

    def __init__(self, question, annotation, raw_sparql, remove_entity_mention, remove_stop_words):
        self.question = question
        self.sparql = SPARQL(raw_sparql)
        self.normalized_question, self.normalized_question_with_numbers, self.lower_indicator = QARow.preprocess(
            self.question, self.sparql.entities, remove_entity_mention, remove_stop_words)
        self.annotation = annotation

    def validate(self):
        return len(self.normalized_question) == len(self.annotation)

    @staticmethod
    def preprocess(line, entities, remove_entity_mention, remove_stop_words):
        line = line.replace('?', ' ').replace('\'', ' ').replace('-', ' ').replace(',', ' ').replace('.', ' ').replace('>', ' ')
        line_split = line.split()
        line_lower = line.lower()
        entity_labels = ''.join([entity.label for entity in entities])
        output, output_with_numbers, lower_indicator = [], [], []
        for word_idx, word in enumerate(line_lower.split()):
            if remove_entity_mention and (word in entity_labels and word not in QARow.stop_words):
                if len(output) == 0 or output[-1] != '<ent>':
                    output.append('<ent>')
                    lower_indicator.append(0)
            elif remove_stop_words and word in QARow.stop_words:
                pass
            elif word.replace('.', '').replace('(', '').isdigit():
                output.append('<num>')
                output_with_numbers.append(word)
                lower_indicator.append(0)
            else:
                output.append(word)
                output_with_numbers.append(word)
                if word == line_split[word_idx]:
                    lower_indicator.append(0)
                else:
                    lower_indicator.append(1)

        return output, output_with_numbers, lower_indicator
