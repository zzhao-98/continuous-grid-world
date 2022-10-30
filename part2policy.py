import math
import os
import random
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F

num_actions = 3
num_states = 4
batch_size = 32
learning_rate = 1e-5        # learning rate
epsilon = 1               # greedy policy
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
        return self.eval_net.state_dict()

    def load_model(self, path):
        self.eval_net, self.target_net = Net(), Net()
        self.eval_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))
        self.eval_net.train()
        self.target_net.train()
        return self.eval_net.state_dict()



www = (10, 40)
os.environ['SDL_VIDEO_WINDOW_POS'] = str(www[0]) + "," + str(www[1])
pygame.init()
white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 200, 0)

# set the window
size = width, height = 500, 500  # window size
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()
gameIcon = pygame.image.load('images/intruder.png')
pygame.display.set_icon(gameIcon)
pygame.display.set_caption('Aircraft Guidance Simulator', 'Spine Runtime')

tick = 30  # update no more than 30 frames in one second
np.set_printoptions(precision=2)

# used for process the output of the neural network. 0: -2, 1: 0, 2: 2
dict_action = {0: -2, 1: 0, 2: 2}

# display the time step at top right corner
def time_display(count):
    font = pygame.font.SysFont("comicsansms", 20)
    text = font.render("Time Step: " + str(count), True, black)
    screen.blit(text, (5, 0))


def reward_display(reward):
    font = pygame.font.SysFont("comicsansms", 20)
    text = font.render("Cum Reward: " + str(reward), True, green)
    screen.blit(text, (5, 25))


# the aircraft object
class DroneSprite(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.src_image = pygame.image.load('images/drone.png')
        self.rect = self.src_image.get_rect()
        self.image = self.src_image
        self.position = (width-50, height-50)  # initial position
        self.speed = 2  # speed of the ownship is 2 pixel per time step
        self.direction = 45
        self.rad = self.direction * math.pi / 180
        # velocity (v_x, v_y) can be decided from the speed scalar and heading angle
        self.vx = -self.speed * math.sin(self.rad)
        self.vy = -self.speed * math.cos(self.rad)
        self.velocity = (self.vx, self.vy)
        # this aircraft will be safe if other aircraft are outside this radius
        self.radius = 16
        # the heading angle will be updated according to this delta_direction.
        self.delta_direction = 0
        # use keyboard to control left or right
        self.keyboard_direction = 0
        self.k_right = 0
        self.k_left = 0

        # cumulative reward of ownship
        self.cumulative_reward = 0

    def update(self, deltat):
        # decide the new heading angle according to the action: self.delta_direction
        self.keyboard_direction = self.k_right + self.k_left
        # if keyboard is controlling, follow the keyboard, else follow the policy
        if self.keyboard_direction:
            self.direction += self.keyboard_direction
        else:
            self.direction += self.delta_direction
        self.direction %= 360  # keep it between (0, 360)
        self.rad = self.direction * math.pi / 180  # turn deg to rad

        # decide the new velocity according to the heading angle
        vx = -self.speed * math.sin(self.rad)
        vy = -self.speed * math.cos(self.rad)
        self.velocity = (vx, vy)

        # decide the new position according to the velocity
        x = self.position[0] + vx
        y = self.position[1] + vy
        self.position = (x, y)
        self.image = pygame.transform.rotate(self.src_image, self.direction)
        self.rect = self.image.get_rect()
        self.rect.center = self.position


# the goal sprite for the ownship
class GoalSprite(pygame.sprite.Sprite):
    def __init__(self, position):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('images/goal.png')
        self.rect = self.image.get_rect()
        self.position = position
        self.radius = 16
        self.rect = self.image.get_rect()
        self.rect.center = self.position

    def update(self):
        self.rect.center = self.position


# get the state vector according to current aircraft position, velocity and goal position
# own is the ownship we are controlling
# goal is the goal sprite
def get_state(own, goal):
    state_list = [own.position[0], own.position[1], goal.position[0],
                  goal.position[1]]

    return np.array(state_list)


# Initialize a DQN model
dqn = DQN()
dqn.load_model('zzh_model_change_state.pt')

time_step = -1  # this is the time step displayed at the top right corner.
# generate a random goal position
goal = GoalSprite((random.random() * width,
                   random.random() * height))

# drone is the ownship we are controlling
drone = DroneSprite()
drone_group = pygame.sprite.RenderPlain(drone)
goal_group = pygame.sprite.RenderPlain(goal)

ep_r = 0 # For getting the total reward for each episode.
simulate = True
# Set the game time up to 1000 iterations.
while simulate:
    # while simulating, you can press esc to exit the pygame window
    # the process will terminate after hitting the boundary or the goal
    time_step += 1

    current_state = get_state(drone, goal)
    action = dqn.choose_action(current_state)
    do_action = dict_action[action]

    drone.delta_direction = do_action


    deltat = clock.tick(tick)
    screen.fill((255, 255, 255))

    # check if the ownship flies out of the map
    if drone.position[0] < 0 or drone.position[0] > width \
            or drone.position[1] < 0 or drone.position[1] > height:
        drone.cumulative_reward += -100
        simulate = False
        print('You hit the wall :(')
        print('Total Reward: ', drone.cumulative_reward)

    # check if the ownship reaches the goal state.
    if pygame.sprite.collide_circle(drone, goal):
        # the ownship reaches the goal position
        collide_goal = True
        drone.cumulative_reward += 500
        simulate = False
        print('You reach the goal!')
        print('Total Reward: ', drone.cumulative_reward)

    # update the drone, goal
    drone_group.update(deltat)
    goal_group.update()

    # draw the aircraft and the goal
    drone_group.draw(screen)
    goal_group.draw(screen)

    # display time steps, number of times hitting wall, goals reached, the cum reward.
    time_display(time_step)
    reward_display(drone.cumulative_reward)

    pygame.display.flip()

    drone.cumulative_reward += -1