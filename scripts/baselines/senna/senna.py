import numpy as np
import jellyfish
import similarity.ngram
from nltk.tag.senna import SennaChunkTagger
from common.linkers.entityOrderedLinker import EntityOrderedLinker
from common.linkers.sorter.embeddingSimilaritySorter import EmbeddingSimilaritySorter
from common.linkers.sorter.stringSimilaritySorter import StringSimilaritySorter
from common.linkers.candidate_generator.elasticCG import ElasticCG
from common.linkers.candidate_generator.elastic import Elastic
from common.linkers.relationOrderLinker import RelationOrderedLinker
from common.dataset.qald_6_ml import Qald_6_ml
from common.dataset.qald_7_ml import Qald_7_ml
from common.dataset.lc_quad import LC_QuAD
from config import config

tagger = SennaChunkTagger('/Users/hamid/workspace/DeepShallowParsingQA/tools/senna')
stop_words = ["a", "as", "able", "about", "above", "according", "accordingly", "across", "actually",
              "after", "afterwards", "again", "against", "aint", "all", "allow", "allows", "almost",
              "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an",
              "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways",
              "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "arent", "around", "as",
              "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "be", "became",
              "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind",
              "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond",
              "both", "brief", "but", "by", "cmon", "cs", "came", "can", "cant", "cannot", "cant", "cause",
              "causes", "certain", "certainly", "changes", "clearly", "co", "com", "come", "comes",
              "concerning", "consequently", "consider", "considering", "contain", "containing", "contains",
              "corresponding", "could", "couldnt", "course", "currently", "definitely", "described",
              "despite", "did", "didnt", "different", "do", "does", "doesnt", "doing", "dont", "done",
              "down", "downwards", "during", "each", "edu", "eg", "eight", "either", "else", "elsewhere",
              "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody",
              "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "far", "few",
              "ff", "fifth", "first", "five", "followed", "following", "follows", "for", "former",
              "formerly", "forth", "four", "from", "further", "furthermore", "get", "gets", "getting",
              "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "had",
              "hadnt", "happens", "hardly", "has", "hasnt", "have", "havent", "having", "he", "hes",
              "hello", "help", "hence", "her", "here", "heres", "hereafter", "hereby", "herein",
              "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully", "how",
              "howbeit", "however", "i", "id", "ill", "im", "ive", "ie", "if", "ignored", "immediate",
              "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar",
              "instead", "into", "inward", "is", "isnt", "it", "itd", "itll", "its", "its", "itself",
              "just", "keep", "keeps", "kept", "know", "knows", "known", "last", "lately", "later",
              "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely",
              "little", "look", "looking", "looks", "ltd", "mainly", "many", "may", "maybe", "me", "mean",
              "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "much", "must", "my",
              "myself", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither",
              "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone",
              "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "obviously", "of", "off",
              "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto", "or",
              "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside",
              "over", "overall", "own", "particular", "particularly", "per", "perhaps", "placed", "please",
              "plus", "possible", "presumably", "probably", "provides", "que", "quite", "qv", "rather",
              "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively",
              "respectively", "right", "said", "same", "saw", "say", "saying", "says", "second",
              "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves",
              "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "should",
              "shouldnt", "since", "six", "so", "some", "somebody", "somehow", "someone", "something",
              "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify",
              "specifying", "still", "sub", "such", "sup", "sure", "ts", "take", "taken", "tell", "tends",
              "th", "than", "thank", "thanks", "thanx", "that", "thats", "thats", "the", "their", "theirs",
              "them", "themselves", "then", "thence", "there", "theres", "thereafter", "thereby",
              "therefore", "therein", "theres", "thereupon", "these", "they", "theyd", "theyll", "theyre",
              "theyve", "think", "third", "this", "thorough", "thoroughly", "those", "though", "three",
              "through", "throughout", "thru", "thus", "to", "together", "too", "took", "toward",
              "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "un", "under",
              "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used",
              "useful", "uses", "using", "usually", "value", "various", "very", "via", "viz", "vs", "want",
              "wants", "was", "wasnt", "way", "we", "wed", "well", "were", "weve", "welcome", "well",
              "went", "were", "werent", "what", "whats", "whatever", "when", "whence", "whenever", "where",
              "wheres", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether",
              "which", "while", "whither", "who", "whos", "whoever", "whole", "whom", "whose", "why",
              "will", "willing", "wish", "with", "within", "without", "wont", "wonder", "would", "would",
              "wouldnt", "yes", "yet", "you", "youd", "youll", "youre", "youve", "your", "yours",
              "yourself", "yourselves", "zero", "whose", "which", "is", ", ", "\\\\", "?", "\\"]


def senna_chunker(text, stop_words=[]):
    result = tagger.tag(text.split())
    phrases = []
    for chunk in result:
        if chunk[1] == 'S-NP':
            if chunk[0].lower() in stop_words:
                phrases.append((chunk[0], "O"))
            else:
                phrases.append((chunk[0], "B-NP"))
        elif chunk[1] == 'B-NP' or chunk[1] == 'I-NP':
            phrases.append((chunk[0], chunk[1]))
        elif chunk[1] == 'E-NP':
            phrases.append((chunk[0], "I-NP"))
        elif chunk[1] == 'S-VP':
            if chunk[0].lower() in stop_words:
                phrases.append((chunk[0], "O"))
            else:
                phrases.append((chunk[0], "B-VP"))
        elif chunk[1] == 'B-VP' or chunk[1] == 'I-VP':
            phrases.append((chunk[0], chunk[1]))
        elif chunk[1] == 'E-VP':
            phrases.append((chunk[0], "I-VP"))
        else:
            phrases.append((chunk[0], "O"))

    return phrases


def get_phrases(sentence):
    parsed = senna_chunker(sentence, stop_words)
    if isinstance(parsed, list):
        phrases = []
        phrase = []
        current_type = ""
        for item in parsed:
            if item[1].startswith("B-NP"):
                phrases.append({"chunk": " ".join(phrase), "class": current_type})
                phrase = [item[0]]
                current_type = "entity"
            elif item[1].startswith("I-NP"):
                phrase.append(item[0])
            if item[1].startswith("B-VP"):
                phrases.append({"chunk": " ".join(phrase), "class": current_type})
                phrase = [item[0]]
                current_type = "relation"
            elif item[1].startswith("I-VP"):
                phrase.append(item[0])
            elif item[1] == "O":
                phrases.append({"chunk": " ".join(phrase), "class": current_type})
                phrase = []
        if len(phrase) > 0:
            phrases.append({"chunk": " ".join(phrase), "class": current_type})
        phrases = [item for item in phrases if len(item["chunk"]) > 1]
        return phrases
    return []


# dataset = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'],
#                   False, False)
# dataset = Qald_7_ml(config['qald_7_ml']['train'], config['qald_7_ml']['test'], config['qald_7_ml']['vocab'],
#                           False, False)
dataset = Qald_6_ml(config['qald_6_ml']['train'], config['qald_6_ml']['test'], config['qald_6_ml']['vocab'],
                    False, False)

if __name__ == '__main__':
    elastic = Elastic(config['elastic']['server'])
    entity_linker = EntityOrderedLinker(
        candidate_generator=ElasticCG(elastic, index_name='entity_whole_match_index'),
        sorters=[StringSimilaritySorter(similarity.ngram.NGram(2).distance, True)],
        vocab=dataset.vocab)

    relation_linker = RelationOrderedLinker(
        candidate_generator=ElasticCG(elastic, index_name='relation_whole_match_index'),
        sorters=[StringSimilaritySorter(jellyfish.levenshtein_distance, False, True),
                 #EmbeddingSimilaritySorter(dataset.word_vectorizer)
                 ],
        vocab=dataset.vocab)

    total_relation_rmm, total_entity_rmm = [], []
    for qarow in dataset.test_set:
        surfaces = get_phrases(qarow.question)
        surfaces = [[item['chunk'].split() for item in surfaces if item['class'] == 'relation'],
                    [item['chunk'].split() for item in surfaces if item['class'] == 'entity']]

        entity_results, entity_score, entity_mrr, found_target_entities = entity_linker.best_ranks(
            list(surfaces[1]), list(surfaces[0]), qarow, 1, False)

        extra_candidates = []
        # extra_candidates.extend(dataset.find_one_hop_relations(found_target_entities))

        relation_results, relation_score, relation_mrr, _ = relation_linker.best_ranks(
            list(surfaces[0]), list(surfaces[1]), qarow, 1, False, extra_candidates)
        total_entity_rmm.append(entity_mrr)
        total_relation_rmm.append(relation_mrr)

    print([np.mean(total_entity_rmm), np.mean(total_relation_rmm)])

#LC-QuAD
# without extra_candiadte
# K=1   [0.2756095238095238, 0.011166666666666665]
# k=10  [0.2812467910396482, 0.024776821789321786]
# with extra_candiadte
#K=1    [0.2756095238095238, 0.09918333333333332]
#K=10   [0.2812467910396482, 0.13607517436267438]

#QALD-7
# with extra_candiadte
#k=1    [0.21400000000000002, 0.0]
#k=10   [0.22224675324675328, 0.006666666666666666]

#QALD-6
# with extra_candiadte
#k=1    [0.22170138888888888, 0.0]
# without extra_candiadte
#k=1    [0.22170138888888888, 0.0026041666666666665]
