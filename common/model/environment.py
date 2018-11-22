import torch


class Environment:
    def __init__(self, linker, positive_reward=1, negative_reward=-0.5):
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.linker = linker
        self.state = []
        self.target = []
        self.input_seq = []
        self.input_seq_size = 0
        self.seq_counter = 0
        self.action_seq = []
        self.num_surface = 0

    def init(self, input_seq):
        self.input_seq = torch.LongTensor(input_seq)
        self.input_seq_size = len(self.input_seq)
        self.seq_counter = 0
        self.state = torch.cat((torch.LongTensor([0]), torch.LongTensor([0]), self.next_token()))
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
        return self.seq_counter == self.input_seq_size + 1 or sum(self.action_seq[-3:]) > 2

    def update_state(self, action, new_token):
        return torch.cat((torch.LongTensor([self.num_surface]), torch.LongTensor([action]), new_token))

    def step(self, action, qarow, k):
        reward = 0
        mrr = 0
        if action == 1:
            if len(self.action_seq) == 0 or self.action_seq[-1] == 0:
                self.num_surface += 1
        self.state = self.update_state(action, self.next_token())
        self.action_seq.append(action)
        is_done = self.is_done()
        if is_done:
            if len(self.action_seq) != len(self.input_seq):
                reward = self.negative_reward
            else:
                last_tag = 0
                surfaces = []
                surface = []
                for idx, tag in enumerate(self.action_seq):
                    if tag == 1:
                        if last_tag == 1:
                            surface.append(self.input_seq[idx])
                        else:
                            surface = [self.input_seq[idx]]
                    elif tag == 0:
                        if len(surface) > 0:
                            surfaces.append(surface)
                            surface = []
                    last_tag = tag
                if len(surface) > 0:
                    surfaces.append(surface)

                score, mrr = self.linker.best_ranks(surfaces, qarow, k)
                reward = score
                if score < 0.6:
                    reward = self.negative_reward
                # else:
                #    reward = score * 10 * self.positive_reward
        return self.state, reward, is_done, mrr
