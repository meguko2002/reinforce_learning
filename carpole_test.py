import gym
import numpy as np

ENV = 'CartPole-v0'
NUM_DIZITIZER = 6
GAMMA = 0.99    # 時間割引率
ETA = 0.5   # 学習係数
MAX_STEPS = 200
NUM_EPISODES = 500

class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_Q_function(self, observation, action, reward, observation_next):
        self.brain.update_Q_table(observation, action, reward, observation_next)

    def get_action(self, observation, episode):
        action = self.brain.decide_action(observation, episode)
        return action

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.q_table = np.random.uniform(low=0, high=1, size=(NUM_DIZITIZER**num_states, num_actions))

    def bins(self, clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def digitize_state(self, observation):
        cart_pos, cart_v, pole_angle, pole_v = observation
        digitized = [np.digitize(cart_pos, bins=self.bins(-2.4, 2.4, NUM_DIZITIZER)),
            np.digitize(cart_v, bins=self.bins(-3.0, 3.0, NUM_DIZITIZER)),
            np.digitize(pole_angle, bins=self.bins(-0.5, 0.5, NUM_DIZITIZER)),
            np.digitize(pole_v, bins=self.bins(-2.0, 2.0, NUM_DIZITIZER)),
            ]
        return sum([x * (NUM_DIZITIZER**i) for i, x in enumerate(digitized)])

    def update_Q_table(self, observation, action, reward, observation_next):
        state = self.digitize_state((observation))
        state_next = self.digitize_state(observation_next)
        Max_Q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = self.q_table[state, action] +\
            ETA * (reward + GAMMA * Max_Q_next - self.q_table[state, action])

    def decide_action(self, observation, episode):
        state = self.digitize_state((observation))
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.rand():
            action = np.argmax(self.q_table[state][:])
        else:
            action = np.random.choice(self.num_actions)
        return action

class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n

        self.agent = Agent(num_states, num_actions)

    def run(self):
        complete_episode = 0
        for episode in range(NUM_EPISODES):
            observation = self.env.reset()

            for step in range(MAX_STEPS):
                if episode % 50 == 0 or episode == NUM_EPISODES-1:
                    self.env.render()
                action = self.agent.get_action(observation, episode)
                observation_next, _, done, _ = self.env.step(action)
                if done:
                    if step < 195:
                        reward = -1
                    else:
                        complete_episode += 1
                        reward = 1
                else:
                    reward = 0
                self.agent.update_Q_function(observation, action, reward, observation_next)
                observation = observation_next
                if done:
                    print('{} Episode: {} times steps  comp:{}'.format(episode, step+1,complete_episode))
                    break


cartpole = Environment()
cartpole.run()
cartpole.env.close()
