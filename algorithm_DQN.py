import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

num_actions = 3
num_states = 4
batch_size = 32
learning_rate = 1e-5        # learning rate
epsilon = 0.9               # greedy policy
discount = 0.9              # reward discount
target_replace_iter = 100   # target update frequency
memory_capacity = 35

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_states, 128)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(128, 16)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(16, num_actions)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value



class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((memory_capacity, num_states * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < epsilon:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] ####Need to optimize
        else:   # random
            action = np.random.randint(0, num_actions)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(memory_capacity, batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :num_states])
        b_a = torch.LongTensor(b_memory[:, num_states:num_states+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, num_states+1:num_states+2])
        b_s_ = torch.FloatTensor(b_memory[:, -num_states:])


        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + discount * q_next.max(1)[0].view(batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        torch.save(self.eval_net.state_dict(), path)

    def load_model(self, path):
        self.eval_net, self.target_net = Net(), Net()
        self.eval_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))
        self.eval_net.train()
        self.target_net.train()

