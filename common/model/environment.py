import torch


class Environment:
    def __init__(self, linker, positive_reward=1, negetive_reward=-0.5):
        self.positive_reward = positive_reward
        self.negetive_reward = negetive_reward
        self.linker = linker
        self.state = []
        self.target = []
        self.input_seq = []
        self.input_seq_embeddings = []
        self.seq_counter = 0
        self.action_seq = []

    def init(self, raw_input, input_seq, input_seq_embeddings):
        self.raw_input = raw_input
        self.input_seq = input_seq
        self.input_seq_embeddings = input_seq_embeddings
        self.input_seq_size = len(self.input_seq_embeddings)
        self.seq_counter = 0
        self.state = torch.cat((torch.FloatTensor([1]), self.next_token()))
        self.action_seq = []

    def next_token(self):
        output = self.input_seq_embeddings[self.seq_counter % self.input_seq_size].reshape(-1)
        self.seq_counter += 1
        return output

    def is_done(self):
        return self.seq_counter == len(self.input_seq_embeddings) + 1  # or np.sum(self.action_seq) > 2

    def update_state(self, action, new_token):
        return torch.cat((torch.FloatTensor([action]), new_token))
        # return torch.cat((torch.FloatTensor([action]), torch.zeros(new_token.size(0)), new_token))

    def step(self, action):
        reward = 0
        self.state = self.update_state(action, self.next_token())
        self.action_seq.append(action)
        is_done = self.is_done()
        if is_done:
            last_tag = 0
            # surfaces = []
            # surface = ''
            # for idx, tag in enumerate(self.action_seq):
            #     if tag == 1:
            #         if last_tag == 1:
            #             surface += ' ' + self.input_seq[idx]
            #         else:
            #             surface = self.input_seq[idx]
            #     elif tag == 0:
            #         if len(surface) > 1:
            #             surfaces.append(surface)
            #             surface = ''
            #     last_tag = tag
            # if len(surface) > 1:
            #     surfaces.append(surface)
            #
            # items = self.linker.link_all(surfaces, self.raw_input)

            if self.action_seq == self.target:
                reward = self.positive_reward
            else:
                reward = self.negetive_reward
        return self.state, reward, is_done

    def set_target(self, target):
        self.target = target
