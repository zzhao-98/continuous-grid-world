import math
import os
import random
import numpy as np
import pygame
from algorithm_DQN import DQN, memory_capacity

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
# training_process = np.array([])
# If want to continue training, uncomment this line.
dqn.load_model('zzh_model_change_state.pt')
training_process = np.load('training_process_change_state.npy')
i_episode = training_process.shape[0] - 1
print('Collecting experience...')
while i_episode < 700:
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
    for sim in range(1000):
        # while simulating, you can press esc to exit the pygame window
        # the process will terminate after hitting the boundary or the goal
        if simulate == False:
            break
        time_step += 1

        current_state = get_state(drone, goal)
        # reward = (width + height - current_state[0])/(width + height)
        # print(current_state)
        action = dqn.choose_action(current_state)
        do_action = dict_action[action]

        drone.delta_direction = do_action


        deltat = clock.tick(tick)
        screen.fill((255, 255, 255))

        # check if the ownship flies out of the map
        if drone.position[0] < 0 or drone.position[0] > width \
                or drone.position[1] < 0 or drone.position[1] > height:
            drone.cumulative_reward += -100
            reward = -2 # Reward set to -2 to touch
            simulate = False

        # check if the ownship reaches the goal state.
        if pygame.sprite.collide_circle(drone, goal):
            # the ownship reaches the goal position
            collide_goal = True
            drone.cumulative_reward += 500
            reward = 2 # Reward set to 2 for reach the goal.
            simulate = False

        # update the drone, goal
        drone_group.update(deltat)
        goal_group.update()
        next_state = get_state(drone, goal)

        distance_goal = np.sqrt((next_state[2] - next_state[0])**2 + (next_state[3] - next_state[1])**2)
        distance_wall = np.min([next_state[0], next_state[1], (width - next_state[0]), (height - next_state[1])])


        # Process the reward.
        reward = 0
        if simulate == True:
            if distance_goal < 150:
                reward += 0.5 # Reward become 1 to approach the goal
            if distance_wall < 70:
                reward += -0.5 # Reward become -1 to aproach the wall
        # Store the transition.
        dqn.store_transition(current_state, action, reward, next_state)

        ep_r += reward
        if dqn.memory_counter > memory_capacity:
            dqn.learn()


        # draw the aircraft and the goal
        drone_group.draw(screen)
        goal_group.draw(screen)

        # display time steps, number of times hitting wall, goals reached, the cum reward.
        time_display(time_step)
        reward_display(drone.cumulative_reward)

        pygame.display.flip()

        drone.cumulative_reward += -1

    i_episode += 1
    # Save the model and the training processing.
    dqn.save_model('zzh_model_change_state.pt')
    print('Episode: ', i_episode, '| Episode_reward: ', round(ep_r, 2))
    training_process = np.append(training_process, round(ep_r, 2))
    np.save("training_process_change_state.npy", training_process)



