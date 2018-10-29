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

    def init(self, input_seq):
        self.input_seq = torch.LongTensor(input_seq)
        self.input_seq_size = len(self.input_seq)
        self.seq_counter = 0
        self.state = torch.cat((torch.LongTensor([0]), self.next_token()))
        self.action_seq = []

    def next_token(self):
        idx = self.seq_counter % self.input_seq_size
        if idx == 0:
            output = torch.cat((torch.LongTensor([0]), self.input_seq[idx].reshape(-1)))
        else:
            output = self.input_seq[idx - 1:idx + 1].reshape(-1)
        self.seq_counter += 1
        return output

    def is_done(self):
        return self.seq_counter == self.input_seq_size + 1  # or np.sum(self.action_seq) > 2

    def update_state(self, action, new_token):
        return torch.cat((torch.LongTensor([action]), new_token))

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
                reward = self.negative_reward
        return self.state, reward, is_done

    def set_target(self, target):
        self.target = target
