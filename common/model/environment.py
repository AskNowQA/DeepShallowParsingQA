import torch
import logging
from common.utils import *
from config import config


class Environment:
    def __init__(self, entity_linker, relation_linker, positive_reward=1, negative_reward=-0.5, dataset=None):
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.entity_linker = entity_linker
        self.relation_linker = relation_linker
        self.dataset = dataset
        self.state, self.target, self.lower_indicator, self.input_seq, self.action_seq, self.split_action_seq = [], [], [], [], [], []
        self.input_seq_size, self.seq_counter, self.num_surface = 0, 0, 0
        self.logger = logging.getLogger('main')
        self.cache = Cache(config['env_cache_path'])

    def init(self, input_seq, lower_indicator):
        self.input_seq = torch.LongTensor(input_seq)
        self.lower_indicator = torch.LongTensor(lower_indicator)
        self.input_seq_size = len(self.input_seq)
        self.seq_counter = 0
        self.state = torch.cat((torch.LongTensor([0, 0]), self.next_token()))
        self.action_seq = []
        self.split_action_seq = []
        self.num_surface = 0

    def next_token(self):
        idx = self.seq_counter % self.input_seq_size
        if idx == 0:
            prev_token = torch.LongTensor([0])
            prev_extra = torch.LongTensor([0])
        else:
            prev_token = self.input_seq[idx - 1].reshape(-1)
            prev_extra = self.lower_indicator[idx - 1].reshape(-1)

        current_token = self.input_seq[idx].reshape(-1)
        current_extra = self.lower_indicator[idx].reshape(-1)
        if idx + 1 == self.input_seq_size:
            next_token = torch.LongTensor([0])
            next_extra = torch.LongTensor([0])
        else:
            next_token = self.input_seq[idx + 1].reshape(-1)
            next_extra = self.lower_indicator[idx + 1].reshape(-1)
        output = torch.cat((prev_extra, current_extra, next_extra, prev_token, current_token, next_token))
        self.seq_counter += 1
        return output

    def is_done(self):
        return self.seq_counter == self.input_seq_size + 1  # or sum(self.action_seq[-3:]) > 2

    @profile
    def update_state(self, action, new_token):
        return torch.cat((torch.LongTensor([self.num_surface, action]), new_token))

    def find_surfaces(self, qarow, split_action_seq):
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
                surface.append(qarow.normalized_question_with_numbers[idx])
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
        self.state = self.update_state(action, self.next_token())
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
                    surfaces, splitted_relations = self.find_surfaces(qarow, self.split_action_seq)

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
                            surfaces_1, splitted_relations_1 = self.find_surfaces(qarow, split_action_seq)
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
