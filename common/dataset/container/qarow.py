from common.dataset.container.sparql import SPARQL
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class QARow:
    stop_words = set(stopwords.words('english'))

    def __init__(self, question, annotation, raw_sparql):
        self.question = question
        self.sparql = SPARQL(raw_sparql)
        self.normalized_question = self.__preprocess(self.question)
        self.annotation = annotation

    def validate(self):
        return len(self.normalized_question) == len(self.annotation)

    def __preprocess(self, line):
        line = line.lower().replace('?', ' ').replace('\'', ' ').replace('-', ' ').replace(',', ' ')
        entity_labels = ''.join([entity.label for entity in self.sparql.entities])
        output = []
        for word in line.split():
            # if word in entity_labels and word not in QARow.stop_words:
            #     if len(output) == 0 or output[-1] != 'ENT':
            #         output.append('ENT')
            # else:
            output.append(word)
        return output
        # word_tokens = word_tokenize(line)
        # filtered_sentence = [w for w in word_tokens if not w in QARow.stop_words]
        # return ' '.join(filtered_sentence)
