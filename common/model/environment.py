import torch
import logging
from common.utils import *
from config import config


class Environment:
    def __init__(self, entity_linker, relation_linker, positive_reward=1, negative_reward=-0.5):
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.entity_linker = entity_linker
        self.relation_linker = relation_linker
        self.state = []
        self.target = []
        self.input_seq = []
        self.input_seq_size = 0
        self.seq_counter = 0
        self.action_seq = []
        self.num_surface = 0
        self.logger = logging.getLogger('main')
        self.cache = Cache(config['env_cache_path'])

    def init(self, input_seq):
        self.input_seq = torch.LongTensor(input_seq)
        self.input_seq_size = len(self.input_seq)
        self.seq_counter = 0
        self.state = torch.cat((torch.LongTensor([0, 0]), self.next_token()))
        self.action_seq = []
        self.num_surface = 0

    def next_token(self):
        idx = self.seq_counter % self.input_seq_size
        if idx == 0:
            prev_token = torch.LongTensor([0])
        else:
            prev_token = self.input_seq[idx - 1].reshape(-1)

        current_token = self.input_seq[idx].reshape(-1)
        if idx + 1 == self.input_seq_size:
            next_token = torch.LongTensor([0])
        else:
            next_token = self.input_seq[idx + 1].reshape(-1)
        output = torch.cat((prev_token, current_token, next_token))
        self.seq_counter += 1
        return output

    def is_done(self):
        return self.seq_counter == self.input_seq_size + 1  # or sum(self.action_seq[-3:]) > 2

    @profile
    def update_state(self, action, new_token):
        return torch.cat((torch.LongTensor([self.num_surface, action]), new_token))

    @profile
    def step(self, action, action_probs, qarow, k, train):
        detailed_rewards = []
        step_reward = 0
        relation_mrr, entity_mrr = 0, 0
        if action > 0:
            if len(self.action_seq) == 0 or self.action_seq[-1] == 0:
                self.num_surface += 1
        self.state = self.update_state(action, self.next_token())
        self.action_seq.append(action)
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
                    last_tag = 0
                    surfaces = [[], []]
                    surface = []
                    for idx, tag in enumerate(self.action_seq):
                        if tag != 0:
                            if last_tag == tag:
                                surface.append(self.input_seq[idx])
                            else:
                                if len(surface) > 0:
                                    surfaces[last_tag - 1].append(surface)
                                surface = [self.input_seq[idx]]
                        elif tag == 0:
                            if len(surface) > 0:
                                if len(surface) > 0:
                                    surfaces[last_tag - 1].append(surface)
                                surface = []
                        last_tag = tag
                    if len(surface) > 0:
                        surfaces[last_tag - 1].append(surface)

                    relation_results, relation_score, relation_mrr = self.relation_linker.best_ranks(list(surfaces[0]),
                                                                                                     qarow, k, train)
                    relation_results = [0 if item < 0.4 else item for item in relation_results]
                    if relation_score < 0.6:
                        relation_score = self.negative_reward
                    entity_results, entity_score, entity_mrr = self.entity_linker.best_ranks(list(surfaces[1]), qarow,
                                                                                             k, train)

                    entity_results = [0 if item < 0.4 else item for item in entity_results]
                    if entity_score < 0.6:
                        entity_score = self.negative_reward
                    step_reward = (relation_score + entity_score) / 2

                    z = 0
                    if step_reward < 0.3:
                        step_reward = self.negative_reward
                    elif step_reward > 0.95:
                        z = 100
                    elif step_reward > 0.9:
                        z = 10
                    # elif step_reward > 0.7:
                    #     z = 3
                    elif step_reward > 0.5:
                        z = 1
                    #     step_reward *= 100

                    rel_idx = 0
                    rel_cntr = 0
                    ent_idx = 0
                    ent_cntr = 0
                    for idx, item in enumerate(self.input_seq):
                        if self.action_seq[idx] == 0:
                            detailed_rewards.append(z)  # len(self.input_seq) * 2.5 / 100
                            pass
                        elif self.action_seq[idx] == 1:
                            detailed_rewards.append(relation_results[rel_idx])
                            rel_cntr += 1
                            if rel_cntr == len(surfaces[0][rel_idx]):
                                rel_idx += 1
                                rel_cntr = 0
                        elif self.action_seq[idx] == 2:
                            detailed_rewards.append(entity_results[ent_idx])
                            ent_cntr += 1
                            if ent_cntr == len(surfaces[1][ent_idx]):
                                ent_idx += 1
                                ent_cntr = 0
                    # detailed_rewards = [0] * len(self.input_seq)
                    # if train:
                    #     self.cache.add(cache_key, (step_reward, mrr))

                self.logger.debug(list(map('{:0.2f}'.format, [entity_mrr, relation_mrr])))
                self.logger.debug('')
        return self.state, detailed_rewards, step_reward, is_done, relation_mrr, entity_mrr
