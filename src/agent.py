import torch
import numpy as np
from collections import deque
from model import Snake_CNN
import random

class SnakeAgent():
    def __init__(self, buffer_size, gamma):
        self.model = Snake_CNN()
        self.target_model = Snake_CNN()
        self.gamma = gamma
        self.erm = ExperienceReplayMemory(buffer_size)
        self.actions = torch.zeros((4))

    def act(self, X, epsilon):
        with torch.no_grad():
            self.actions = self.model(X.reshape((1, X.shape[0], X.shape[1], X.shape[2])))[0]
        if np.random.random() > epsilon:
            direction = torch.argmax(self.actions).item()
        else:
            direction = np.random.randint(0, 4)
        return direction
    
    def train(self, batch_size):
        states, actions, rewards, next_states, dones = self.erm.sample(batch_size)
        if states is None:
            return

        states = torch.tensor(states.detach(), dtype = torch.float32)
        rewards = torch.tensor(rewards, dtype = torch.float32)
        next_states = torch.tensor(next_states.detach(), dtype = torch.float32)
        q_values = self.model(states).detach().numpy()
        #targets = torch.tensor(q_values, dtype=torch.float32)
        targets = np.zeros([batch_size, 4])

        with torch.no_grad():
            next_max_q_values = torch.max(self.target_model(next_states), 1)
        for i in range(batch_size):
            if dones[i] == True:
                next_max_q_values[0][i] = 0.#-1.0
            targets[i] = q_values[i]
            targets[i, actions[i]]  = rewards[i] + self.gamma * next_max_q_values[0][i]
        
        targets = torch.tensor(targets, dtype=torch.float32) 
        loss = self.model.fit(states, targets)
        return loss
        return
        states, actions, rewards, next_states, dones = self.erm.sample(batch_size)
        if states is None:
            return
        with torch.no_grad():
            q_values = self.model(states)
            next_q_values = self.target_model(next_states)
        next_max_q_values = torch.zeros((states.shape[0], 1))
        targets = torch.zeros(q_values.shape)
        for i in range(states.shape[0]):

            if dones[i] == True:
                next_max_q_values[i] = 0.#-1-
            else:
                next_max_q_values[i] = torch.max(next_q_values[i])

            targets[i, actions[i]] += rewards[i] + self.gamma * next_max_q_values[i] - q_values[i, actions[i]]

        self.optimizer.zero_grad()
        self.model.train()

        predictions = self.model(states)
        loss = self.criterion(predictions, targets)
        
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
class ExperienceReplayMemory():
    def __init__(self, size):
        self.states = None
        self.directions = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None
        self.max_size = size
        self.size = 0
        self.i = 0

    def append(self, state, action, reward, next_state, done, store_zero_reward_prop = 1.):
        if reward == 0 and np.random.random() > store_zero_reward_prop:
            return

        if self.states is None:
            self.states = torch.zeros((self.max_size, state.shape[0], state.shape[1], state.shape[2]))
        self.states[self.i] = state
        if self.actions is None:
            self.actions = torch.zeros((self.max_size, 1), dtype=torch.int32)
        self.actions[self.i] = action
        if self.rewards is None:
            self.rewards = torch.zeros((self.max_size, 1))
        self.rewards[self.i] = reward
        if self.next_states is None:
            self.next_states = torch.zeros((self.max_size, next_state.shape[0], next_state.shape[1], next_state.shape[2]))
        self.next_states[self.i] = next_state
        if self.dones is None:
            self.dones = torch.zeros((self.max_size, 1))
        self.dones[self.i] = 1 if done else 0

        if self.size < self.max_size:
            self.size += 1
        self.i = self.i+1 if self.i<self.max_size-1 else 0

    def sample(self, batch_size):
        if len(self.states) < batch_size:
            return None, None, None, None, None
        
        indices = torch.randint(0, self.size, (batch_size,))
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]

        return states, actions, rewards, next_states, dones