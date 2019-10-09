import torch
import logging
from common.utils import *
from config import config
from common.dataset.container.uri import URI


class Environment:
    def __init__(self, entity_linker, relation_linker, positive_reward=1, negative_reward=-0.5, dataset=None, b=1):
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.entity_linker = entity_linker
        self.relation_linker = relation_linker
        self.dataset = dataset
        self.state, self.target, self.lower_indicator, self.input_seq, self.action_seq, self.split_action_seq = [], [], [], [], [], []
        self.input_seq_size, self.seq_counter, self.num_surface = 0, 0, 0
        self.logger = logging.getLogger('main')
        self.cache = Cache(config['env_cache_path'])
        self.b = b

    def init(self, input_seq, lower_indicator):
        self.input_seq = torch.LongTensor(input_seq)
        self.lower_indicator = torch.LongTensor(lower_indicator)
        self.input_seq_size = len(self.input_seq)
        self.seq_counter = 0
        self.state = torch.cat((torch.LongTensor([0, 0]), self.next_token(self.b)))
        self.action_seq = []
        self.split_action_seq = []
        self.num_surface = 0

    def next_token(self, b):
        idx = self.seq_counter % self.input_seq_size
        prev_tokens = torch.LongTensor([0] * b)
        prev_extras = torch.LongTensor([0] * b)
        p_i = 0
        for i in range(idx - b, idx):
            if i >= 0:
                prev_tokens[p_i] = self.input_seq[i].reshape(-1)
                prev_extras[p_i] = self.lower_indicator[i].reshape(-1)
            p_i += 1

        current_token = self.input_seq[idx].reshape(-1)
        current_extra = self.lower_indicator[idx].reshape(-1)
        n_i = 0
        next_tokens = torch.LongTensor([0] * b)
        next_extras = torch.LongTensor([0] * b)
        for i in range(idx, idx + b):
            if i < self.input_seq_size - 1:
                next_tokens[n_i] = self.input_seq[i + 1].reshape(-1)
                next_extras[n_i] = self.lower_indicator[i + 1].reshape(-1)
            n_i += 1
        output = torch.cat((prev_extras, current_extra, next_extras, prev_tokens, current_token, next_tokens))
        self.seq_counter += 1
        return output

    def is_done(self):
        return self.seq_counter == self.input_seq_size + 1  # or sum(self.action_seq[-3:]) > 2

    @profile
    def update_state(self, action, new_token):
        return torch.cat((torch.LongTensor([self.num_surface, action]), new_token))

    def find_surfaces(self, normalized_question_with_numbers, split_action_seq):
        last_tag = 0
        surfaces = [[], []]
        surface = []
        splitted_relations = []
        for idx, tag in enumerate(self.action_seq):
            tag = int(tag)
            if tag != 0:
                if last_tag != tag or (last_tag == tag and split_action_seq[idx] == 0):
                    if len(surface) > 0:
                        surfaces[last_tag - 1].append(surface)
                        if last_tag == tag and split_action_seq[idx] == 0:
                            splitted_relations.append(idx)
                    surface = []
                surface.append(normalized_question_with_numbers[idx])
            elif tag == 0:
                if len(surface) > 0:
                    surfaces[last_tag - 1].append(surface)
                    surface = []
            last_tag = tag
        if len(surface) > 0:
            surfaces[last_tag - 1].append(surface)
        return surfaces, splitted_relations

    @profile
    def step(self, action, action_probs, split_action, qarow, k, train):
        detailed_rewards, split_action_target = [], []
        step_reward, relation_mrr, entity_mrr = 0, 0, 0
        if action > 0:
            if len(self.action_seq) == 0 or self.action_seq[-1] == 0:
                self.num_surface += 1
        self.state = self.update_state(action, self.next_token(self.b))
        self.action_seq.append(action)
        self.split_action_seq.append(split_action)

        is_done = self.is_done()
        if is_done:
            self.logger.debug(qarow.question)
            if self.logger.level == logging.DEBUG:
                for word, prob in zip(qarow.normalized_question, action_probs):
                    Utils.print_color(word, bg=Utils.rgb(*prob), end=' ')
                print()
            self.logger.debug(list(zip(qarow.normalized_question,
                                       [['{:0.2f}'.format(item) for item in probs] for probs in action_probs])))
            if len(self.action_seq) != len(self.input_seq):
                step_reward = self.negative_reward
            else:
                cache_key = qarow.question + ''.join(map(str, map(int, self.action_seq)))
                if train and self.cache.has(cache_key):
                    step_reward, mrr = self.cache.get(cache_key)
                else:
                    surfaces, splitted_relations = self.find_surfaces(qarow.normalized_question_with_numbers,
                                                                      self.split_action_seq)

                    extra_candidates = []
                    entity_results, entity_score, entity_mrr, found_target_entities = self.entity_linker.best_ranks(
                        list(surfaces[1]), list(surfaces[0]), qarow, k, train)

                    if not train:
                        extra_candidates.extend(self.dataset.find_one_hop_relations(found_target_entities))

                    relation_results, relation_score, relation_mrr, _ = self.relation_linker.best_ranks(
                        list(surfaces[0]), list(surfaces[1]), qarow, k, train, extra_candidates)

                    split_action_target = list(self.split_action_seq)
                    if train:
                        for item in splitted_relations:
                            split_action_seq = list(self.split_action_seq)
                            split_action_seq[item] = 1
                            surfaces_1, splitted_relations_1 = self.find_surfaces(
                                qarow.normalized_question_with_numbers, split_action_seq)
                            relation_results_1, relation_score_1, relation_mrr_1, _ = self.relation_linker.best_ranks(
                                list(surfaces_1[0]), [], qarow, k, train)
                            if relation_score_1 > relation_score:
                                split_action_target[item] = 1
                            else:
                                split_action_target[item] = 0

                    relation_results = [[item - 0.5 if item < 0.5 else item for item in items] for items in
                                        relation_results]
                    entity_results = [[item - 0.5 if item < 0.5 else item for item in items] for items in
                                      entity_results]
                    step_reward = (relation_score + entity_score) / 2
                    step_reward -= 0.5

                    rel_idx, rel_cntr, ent_idx, ent_cntr = 0, 0, 0, 0
                    for idx, item in enumerate(self.input_seq):
                        if self.action_seq[idx] == 0:
                            detailed_rewards.append(0.1)  # -0.01 general # 0.1 only embedding lc_quad
                            pass
                        elif self.action_seq[idx] == 1:
                            detailed_rewards.append(relation_results[rel_idx][rel_cntr])
                            rel_cntr += 1
                            if rel_cntr == len(relation_results[rel_idx]):
                                rel_idx += 1
                                rel_cntr = 0
                        elif self.action_seq[idx] == 2:
                            detailed_rewards.append(entity_results[ent_idx][ent_cntr])
                            ent_cntr += 1
                            if ent_cntr == len(entity_results[ent_idx]):
                                ent_idx += 1
                                ent_cntr = 0

                    # if train:
                    #     self.cache.add(cache_key, (step_reward, mrr))

                self.logger.debug(list(map('{:0.2f}'.format, [entity_mrr, relation_mrr])))
                self.logger.debug('')
                # detailed_rewards = [0 for r in detailed_rewards]
        return self.state, detailed_rewards, step_reward, split_action_target, is_done, relation_mrr, entity_mrr

    def connecting_relations_offset(self, candidate_entities, question, offset):
        try:
            c1 = candidate_entities[0]['uris'][0]['confidence'] - offset
            c2 = candidate_entities[1]['uris'][0]['confidence'] - offset
            candidate_relations = []
            for first_entity in candidate_entities[0]['uris']:
                if first_entity['confidence'] < c1:
                    break
                for second_entity in candidate_entities[1]['uris']:
                    if second_entity['confidence'] < c2:
                        break
                    candidate_relations.append(
                        Utils.relations_connecting_entities(first_entity['uri'], second_entity['uri'], 'q.cache'))

            candidate_relations = [set([t for item in candidate_relations for t in item[0] if len(t) > 0]),
                                   set([t for item in candidate_relations for t in item[1] if len(t) > 0])]
            relations_sim, rel_sims = {}, []
            for rel_id, relations in enumerate(candidate_relations):
                rel_sims.append({})
                relations_sim[rel_id] = [0, '']
                uris = [URI(item) for item in relations]
                for uri in uris:
                    uri.coded = self.dataset.decode(uri)
                uris = [[item.raw_uri, item.label] + list(item.coded) for item in uris]
                for word in question.split():
                    word_relation_similarity = self.relation_linker.sorters[1].sort(word, question, uris)
                    if len(word_relation_similarity) > 0:
                        best_score = relations_sim[rel_id][0]
                        if word_relation_similarity[0][4] > best_score:
                            relations_sim[rel_id] = [word_relation_similarity[0][4], word]
                        for item in word_relation_similarity:
                            score = 0
                            if item[0] in rel_sims[rel_id]:
                                score = rel_sims[rel_id][item[0]]
                            if item[4] > score:
                                rel_sims[rel_id][item[0]] = item[4]
                if len(rel_sims[rel_id]) != len(uris):
                    for uri in uris:
                        if uri[0] not in rel_sims[rel_id]:
                            rel_sims[rel_id][uri[0]] = 0.1
            return [[item[1][1]] for item in relations_sim.items()], [
                {'surface': [question.index(relations_sim[rel_id][1]), len(relations_sim[rel_id][1])],
                 'uris': [{'uri': item, 'confidence': rels[item]} for item in rels]} for rel_id, rels in
                enumerate(rel_sims)]
        except:
            return None, None

    def connecting_relation_offset(self, candidate_entities, question, offset):
        try:
            c1 = candidate_entities[0]['uris'][0]['confidence'] - offset
            c2 = candidate_entities[1]['uris'][0]['confidence'] - offset
            candidate_relations = []
            for first_entity in candidate_entities[0]['uris']:
                if first_entity['confidence'] < c1:
                    break
                for second_entity in candidate_entities[1]['uris']:
                    if second_entity['confidence'] < c2:
                        break
                    try:
                        relations = Utils.relation_connecting_entities(
                            first_entity['uri'], second_entity['uri'], 'q2.cache')
                        if len(relations) > 0:
                            candidate_relations.append(relations)
                    except:
                        pass
            if len(candidate_relations) == 0:
                return None, None
            relations_sim, rel_sims = {}, []
            for rel_id, relations in enumerate(candidate_relations):
                rel_sims.append({})
                relations_sim[rel_id] = [0, '']
                uris = [URI(item) for item in relations]
                for uri in uris:
                    uri.coded = self.dataset.decode(uri)
                uris = [[item.raw_uri, item.label] + list(item.coded) for item in uris]
                for word in question.split():
                    word_relation_similarity = self.relation_linker.sorters[1].sort(word, question, uris)
                    if len(word_relation_similarity) > 0:
                        best_score = relations_sim[rel_id][0]
                        if word_relation_similarity[0][4] > best_score:
                            relations_sim[rel_id] = [word_relation_similarity[0][4], word]
                        for item in word_relation_similarity:
                            score = 0
                            if item[0] in rel_sims[rel_id]:
                                score = rel_sims[rel_id][item[0]]
                            if item[4] > score:
                                rel_sims[rel_id][item[0]] = item[4]
                if len(rel_sims[rel_id]) == 0:
                    for uri in uris:
                        rel_sims[rel_id][uri[0]] = 0.1
            return [[item[1][1]] for item in relations_sim.items()], [
                {'surface': [question.index(relations_sim[rel_id][1]), len(relations_sim[rel_id][1])],
                 'uris': [{'uri': item, 'confidence': rels[item]} for item in rels]} for rel_id, rels in
                enumerate(rel_sims)]
        except:
            return None, None

    def connecting_relations(self, fn, candidate_entities, question):
        offset = 0
        while True:
            surfaces_relations, candidate_rels = fn(candidate_entities, question, offset)
            offset += 0.1
            if offset > 0.4:
                break
            if all([''.join(s) == '' for s in surfaces_relations]):
                continue
            if candidate_rels is not None and all([len(item['uris']) > 0 for item in candidate_rels]):
                break
        for rel in candidate_rels:
            if len(rel['uris']) > 10:
                rel['uris'] = [item for item in rel['uris'] if item['confidence'] > 0.1]
        return surfaces_relations, candidate_rels

    def link(self, action, split_action, k, question, normalized_question_with_numbers, connecting_relations,
             free_relation_match, connecting_relation):
        if action > 0:
            if len(self.action_seq) == 0 or self.action_seq[-1] == 0:
                self.num_surface += 1
        self.state = self.update_state(action, self.next_token(self.b))
        self.action_seq.append(action)
        self.split_action_seq.append(split_action)

        is_done = self.is_done()
        result = {}
        if is_done:

            if len(self.action_seq) == len(self.input_seq):
                surfaces, splitted_relations = self.find_surfaces(normalized_question_with_numbers,
                                                                  self.split_action_seq)

                if len(surfaces[0]) > 0 and max([len(item) for item in surfaces[0]]) > 2:
                    for idx, item in enumerate(surfaces[0]):
                        if len(item) > 2:
                            surfaces[0].append([item[-1]])
                            surfaces[0][idx] = item[:2]
                candidate_entities, top_candidate_entities = self.entity_linker.ranked_link(
                    list(surfaces[1]), list(surfaces[0]), question, k, extra_candidates=None)

                candidate_relations = None
                if connecting_relation and len(candidate_entities) == 2:
                    surfaces_relations, candidate_relations = self.connecting_relations(
                        self.connecting_relation_offset, candidate_entities, question)
                    if candidate_relations is not None:
                        surfaces[0] = surfaces_relations
                if connecting_relations and len(candidate_entities) == 2:
                    surfaces_relations, candidate_relations = self.connecting_relations(
                        self.connecting_relations_offset, candidate_entities, question)
                    if candidate_relations is not None:
                        surfaces[0] = surfaces_relations
                if candidate_relations is None:
                    extra_candidates_flatted = self.dataset.find_one_hop_relations(top_candidate_entities)

                    candidate_relations_1, _ = self.relation_linker.ranked_link(
                        list(surfaces[0]), list(surfaces[1]), question, k, extra_candidates_flatted)
                    if len(surfaces[0]) > 0 and len(extra_candidates_flatted) > 0 and free_relation_match:
                        candidate_relations_2, _ = self.relation_linker.ranked_link(
                            list(surfaces[0]), list(surfaces[1]), question, k, None)
                        candidate_relations = []
                        for item1, item2 in zip(candidate_relations_1, candidate_relations_2):
                            candidate_relations.append(
                                {'surface': item1['surface'], 'uris': item1['uris'] + item2['uris']})
                    else:
                        candidate_relations = candidate_relations_1

                chunks = [[item, 'entity'] for item in surfaces[1]] + [[item, 'relation'] for item in surfaces[0]]

                result = {'chunks': [{'chunk': ' '.join(item[0]), 'class': item[1]} for item in chunks],
                          'entities': candidate_entities,
                          'relations': candidate_relations}

        return self.state, is_done, result
