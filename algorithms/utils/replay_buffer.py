from collections import namedtuple, deque
import random
import torch

Transition = namedtuple("Transition", "state action reward next_state done")


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, k):
        batch = random.sample(self.buffer, k)
        batch = Transition(*zip(*batch))

        state = torch.stack(batch.state)
        action = torch.stack(batch.action)
        reward = torch.stack(batch.reward)
        next_state = torch.stack(batch.next_state)
        done = torch.stack(batch.done)

        return state, action, reward, next_state, done
