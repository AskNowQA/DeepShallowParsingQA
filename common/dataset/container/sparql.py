import re


class SPARQL:
    def __init__(self, raw_sparql):
        self.raw_sparql = raw_sparql
        self.relations = self.__extract_relations(self.raw_sparql)

    def __extract_relations(self, sparql):
        output = re.findall('<[^>]*>', sparql)
        output = [item.strip('<>') for item in output if '/ontology/' in item or '/property/' in item]
        return output
