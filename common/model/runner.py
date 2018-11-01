class Runner:
    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent

    # @profile
    def step(self, input, qarow, e, train=True):
        rewards = []
        action_log_probs = []
        total_reward = []
        running_reward = 0
        self.environment.init(input)
        state = self.environment.state
        while True:
            action_dist, action, action_log_prob = self.agent.select_action(state, e)
            new_state, reward, done = self.environment.step(action, qarow)
            running_reward += reward
            rewards.append(reward)
            action_log_probs.append(action_log_prob)
            state = new_state
            if done:
                if train:
                    self.agent.optimize(rewards, action_log_probs)
                total_reward.append(running_reward)
                break
        return total_reward
