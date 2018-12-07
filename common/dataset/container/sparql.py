import re
from common.dataset.container.uri import URI


class SPARQL:
    def __init__(self, raw_sparql):
        self.raw_sparql = raw_sparql
        self.entities, self.relations = self.__extract_relations(self.raw_sparql)

    def __extract_relations(self, sparql):
        output = re.findall('<[^>]*>', sparql)
        relations = set([item for item in output if '/ontology/' in item or '/property/' in item])
        relations = [URI(item) for item in relations]
        entities = [URI(item, ngram=3) for item in output if '/resource/' in item]
        return entities, relations
