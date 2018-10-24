class Runner:
    def __init__(self, environment, agent, word_vectorizer):
        self.environment = environment
        self.agent = agent
        self.word_vectorizer = word_vectorizer

    def step(self, raw_input, input, e):
        rewards = []
        action_log_probs = []
        total_reward = []
        running_reward = 0
        self.environment.init(raw_input, input.split(), self.word_vectorizer.decode(input))
        state = self.environment.state
        while True:
            action_dist, action, action_log_prob = self.agent.select_action(state, e)
            new_state, reward, done = self.environment.step(action)
            running_reward += reward
            rewards.append(reward)
            action_log_probs.append(action_log_prob)
            state = new_state
            if done:
                self.agent.optimize(rewards, action_log_probs)
                total_reward.append(running_reward)
                break
        return total_reward
