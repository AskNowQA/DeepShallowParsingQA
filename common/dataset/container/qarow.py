from common.dataset.container.sparql import SPARQL


class QARow:
    def __init__(self, question, annotation, raw_sparql):
        self.question = question
        self.normalized_question = self.__preprocess(self.question)
        self.annotation = annotation
        self.sparql = SPARQL(raw_sparql)

    def validate(self):
        return len(self.normalized_question.split()) == len(self.annotation)

    def __preprocess(self, line):
        return line.lower().replace('?', ' ').replace('\'', ' ').replace('-', ' ').replace(',', ' ')
