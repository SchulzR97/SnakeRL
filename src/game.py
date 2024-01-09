import cv2 as cv
import torch
import numpy as np

NUMBER_SNAKE_HEAD = 1
NUMBER_SNAKE = -1
NUMBER_BOARDER = -2
NUMBER_FOOD = 2
REWARD_DEFAULT = 0.01
REWARD_FOOD = 1.
REWARD_COLLISSION = -1.
DEBUG = False
MAX_STEPS = 100
BORDER = 1

class SnakeBoard():
    def __init__(self, size = (15, 15)):
        self.size = (size[0]+BORDER*2, size[1]+BORDER*2)
        self.max_score = 0
        self.scores = []
        self.score = 0
        self.max_score = 0
        self.reset()

    def reset(self):
        self.steps = 0
        if self.score > self.max_score:
            self.max_score = self.score
        if self.score is np.NAN:
            self.scores.append(0)
        else:
            self.scores.append(self.score)
        self.done = False
        self.score = 0
        self.last_dist = 0
        self.last_reward = 0
        self.reward_cum = 0
        self.state = torch.zeros(self.size)
        self.state[:BORDER,:] = NUMBER_BOARDER
        self.state[-BORDER:,:] = NUMBER_BOARDER
        self.state[:,:BORDER] = NUMBER_BOARDER
        self.state[:,-BORDER:] = NUMBER_BOARDER
        self.snake = []

        snake_row = np.random.randint(BORDER+3, self.size[0]-BORDER-3)
        snake_col = np.random.randint(BORDER+3, self.size[1]-BORDER-3)

        self.direction = np.random.randint(0, 4)
        
        self.state[snake_row, snake_col] = NUMBER_SNAKE

        if self.direction == 0:         # top
            self.state[snake_row:snake_row+3, snake_col] = NUMBER_SNAKE
            self.state[snake_row, snake_col] = NUMBER_SNAKE_HEAD
            self.snake.append([snake_row+0, snake_col+0])
            self.snake.append([snake_row+1, snake_col+0])
            self.snake.append([snake_row+2, snake_col+0])
        if self.direction == 1:         # right
            self.state[snake_row, snake_col-2:snake_col] = NUMBER_SNAKE
            self.state[snake_row, snake_col] = NUMBER_SNAKE_HEAD
            self.snake.append([snake_row+0, snake_col+0])
            self.snake.append([snake_row+0, snake_col-1])
            self.snake.append([snake_row+0, snake_col-2])
        if self.direction == 2:         # bottom
            self.state[snake_row-2:snake_row, snake_col] = NUMBER_SNAKE
            self.state[snake_row, snake_col] = NUMBER_SNAKE_HEAD
            self.snake.append([snake_row+0, snake_col+0])
            self.snake.append([snake_row-1, snake_col+0])
            self.snake.append([snake_row-2, snake_col+0])
        if self.direction == 3:         # left
            self.state[snake_row, snake_col:snake_col+3] = NUMBER_SNAKE
            self.state[snake_row, snake_col] = NUMBER_SNAKE_HEAD
            self.snake.append([snake_row+0, snake_col+0])
            self.snake.append([snake_row+0, snake_col+1])
            self.snake.append([snake_row+0, snake_col+2])

        self.place_food()

    def place_food(self):
        while True:
            food_row = np.random.randint(0, self.size[0])
            food_col = np.random.randint(0, self.size[1])
            if self.state[food_row, food_col] == 0:
                break

        self.food_pos = [food_row, food_col]
        self.state[food_row, food_col] = NUMBER_FOOD

    def step_direction(self, direction):
        action = 0
        if self.direction == 0:     # top
            if direction == 0:
                action = 0
            if direction == 1:
                action = 1
            if direction == 2:
                self.last_reward = REWARD_COLLISSION
                self.reward_cum += REWARD_COLLISSION
                self.steps += 1
                self.done = True
                return REWARD_COLLISSION
            if direction == 3:
                action = -1
        if self.direction == 1:     # right
            if direction == 0:
                action = -1
            if direction == 1:
                action = 0
            if direction == 2:
                action = 1
            if direction == 3:
                self.last_reward = REWARD_COLLISSION
                self.reward_cum += REWARD_COLLISSION
                self.steps += 1
                self.done = True
                return REWARD_COLLISSION
        if self.direction == 2:     # bottom
            if direction == 0:
                self.last_reward = REWARD_COLLISSION
                self.reward_cum += REWARD_COLLISSION
                self.steps += 1
                self.done = True
                return REWARD_COLLISSION
            if direction == 1:
                action = -1
            if direction == 2:
                action = 0
            if direction == 3:
                action = 1
        if self.direction == 3:     # left
            if direction == 0:
                action = 1
            if direction == 1:
                self.last_reward = REWARD_COLLISSION
                self.reward_cum += REWARD_COLLISSION
                self.steps += 1
                self.done = True
                return REWARD_COLLISSION
            if direction == 2:
                action = -1
            if direction == 3:
                action = 0

        return self.step(action)

    def step(self, action):
        if self.steps >= MAX_STEPS:
            self.done = True
            self.last_dist = REWARD_COLLISSION
            return REWARD_COLLISSION
        if self.direction == 0:     # top
            if action == -1:        # turn left
                self.direction = 3  # left
            if action == 1:         # turn right
                self.direction = 1  # right
        elif self.direction == 1:     # right
            if action == -1:        # turn left
                self.direction = 0  # top
            if action == 1:         # turn right
                self.direction = 2  # bottom
        elif self.direction == 2:     # bottom
            if action == -1:        # turn left
                self.direction = 1  # right
            if action == 1:         # turn right
                self.direction = 3  # left
        elif self.direction == 3:     # left
            if action == -1:        # turn left
                self.direction = 2  # bottom
            if action == 1:         # turn right
                self.direction = 0  # top

        reward = self.move_snake()
        self.last_reward = reward
        self.reward_cum += reward
        self.steps += 1
        return reward

    def move_snake(self):
        if self.check_collission():
            self.done = True
            return REWARD_COLLISSION

        tail = [self.snake[-1][0], self.snake[-1][1]]

        for i in reversed(range(len(self.snake))):
            if i > 0:
                self.snake[i] = self.snake[i-1]
        
        if self.direction == 0:     # top
            self.snake[0] = [self.snake[1][0]-1, self.snake[1][1]]
        if self.direction == 1:     # right
            self.snake[0] = [self.snake[1][0], self.snake[1][1]+1]
        if self.direction == 2:     # bottom
            self.snake[0] = [self.snake[1][0]+1, self.snake[1][1]]
        if self.direction == 3:     # left
            self.snake[0] = [self.snake[1][0], self.snake[1][1]-1]

        if self.state[self.snake[0][0], self.snake[0][1]] == NUMBER_FOOD:
            self.snake.append(tail)
            self.state[self.snake[0][0], self.snake[0][1]] = NUMBER_SNAKE_HEAD
            self.state[self.snake[1][0], self.snake[1][1]] = NUMBER_SNAKE
            self.place_food()
            self.steps = 0
            self.score += 1
            return REWARD_FOOD
        else:
            self.state[self.snake[0][0], self.snake[0][1]] = NUMBER_SNAKE_HEAD
            self.state[tail[0], tail[1]] = 0
        self.state[self.snake[1][0], self.snake[1][1]] = NUMBER_SNAKE

        dist = np.sqrt((self.food_pos[0] - self.snake[0][0])**2+(self.food_pos[1] - self.snake[0][1])**2)
        dist_max = np.sqrt((self.size[0]-2)**2+(self.size[1]-2)**2)
        dist_rel = dist / dist_max

        #reward = (self.last_dist - dist_rel) * REWARD_FOOD * 4.
        self.last_dist = dist_rel
        return REWARD_DEFAULT
    
    def check_collission(self):
        if self.direction == 0:     # top
            #if self.state[]
            if self.snake[0][0] - 1 < BORDER:
                return True    # wall collission
            if self.state[self.snake[0][0]-1, self.snake[0][1]] == NUMBER_SNAKE:
                return True    # self collission
        if self.direction == 1:     # right
            if self.snake[0][1] + 1 >= self.size[1]-BORDER:
                return True    # wall collission
            if self.state[self.snake[0][0], self.snake[0][1]+1] == NUMBER_SNAKE:
                return True    # self collission
        if self.direction == 2:     # bottom
            if self.snake[0][0] + 1 >= self.size[0]-BORDER:
                return True    # wall collission
            if self.state[self.snake[0][0]+1, self.snake[0][1]] == NUMBER_SNAKE:
                return True    # self collission
        if self.direction == 3:     # left
            if self.snake[0][1] - 1 < BORDER:
                return True    # wall collission
            if self.state[self.snake[0][0], self.snake[0][1]-1] == NUMBER_SNAKE:
                return True    # self collission
        return False

    def render(self, actions = None):
        grid_size = 40

        img = np.full((self.size[0]*grid_size, self.size[1]*grid_size, 3), 1.)

        for r in range(self.size[0]):
            for c in range(self.size[1]):
                img = cv.rectangle(img, (c*grid_size, r*grid_size), ((c+1)*grid_size, (r+1)*grid_size), (0.95,0.95,0.95), 1)
                if self.state[r, c] == NUMBER_SNAKE:
                    img = cv.rectangle(img, (c*grid_size, r*grid_size), ((c+1)*grid_size, (r+1)*grid_size), (65/255,127/255,70/255), -1)
                if self.state[r, c] == NUMBER_BOARDER:
                    img = cv.rectangle(img, (c*grid_size, r*grid_size), ((c+1)*grid_size, (r+1)*grid_size), (50/255,50/255,50/255), -1)
                if self.state[r, c] == NUMBER_SNAKE_HEAD:
                    img = cv.rectangle(img, (c*grid_size, r*grid_size), ((c+1)*grid_size, (r+1)*grid_size), (35/255,92/255,40/255), -1)
                    #img = cv.circle(img, (int(c*grid_size + 0.5*grid_size), int(r*grid_size + 0.5 * grid_size)), int(0.8*grid_size), (35/255,92/255,40/255), -1)
                if self.state[r, c] == NUMBER_FOOD:
                    img = cv.rectangle(img, (c*grid_size, r*grid_size), ((c+1)*grid_size, (r+1)*grid_size), (66/255,245/255,81/255), -1)

        if actions is not None:
            actions = (actions - actions.min()) / (actions.max() - actions.min())
            for i, val in enumerate(actions):
                #if self.direction == 0:
                if i == 0:
                    r = self.snake[0][0]-1
                    c = self.snake[0][1]
                    pt2 = (c*grid_size+int(0.5*grid_size), r*grid_size+int(0.5*grid_size)-int(val.item()*grid_size))
                if i == 1:
                    r = self.snake[0][0]
                    c = self.snake[0][1]+1
                    pt2 = (c*grid_size+int(0.5*grid_size)+int(val.item()*grid_size), r*grid_size+int(0.5*grid_size))
                if i == 2:
                    r = self.snake[0][0]+1
                    c = self.snake[0][1]
                    pt2 = (c*grid_size+int(0.5*grid_size), r*grid_size+int(0.5*grid_size)+int(val.item()*grid_size))
                if i == 3:
                    r = self.snake[0][0]
                    c = self.snake[0][1]-1
                    pt2 = (c*grid_size+int(0.5*grid_size)-int(val.item()*grid_size), r*grid_size+int(0.5*grid_size))
            
                color = (0.,val.item(), (1.-val.item()))
                #img = cv.rectangle(img, (c*grid_size, r*grid_size), ((c+1)*grid_size, (r+1)*grid_size), color, -1)
                img = cv.arrowedLine(img, (self.snake[0][1]*grid_size+int(0.5*grid_size), self.snake[0][0]*grid_size+int(0.5*grid_size)), pt2, color, 5, tipLength=0.1)

        img = cv.putText(img, "SCORE: {0:0.0f}".format(self.score), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0.9, 0.9, 0.9), 1)

        if DEBUG:
            img = cv.putText(img, "FOOD: {0}".format(self.food_pos), (70, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0., 0., 0.), 1)
            img = cv.putText(img, "DIRECTION: {0}".format(self.direction), (70, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0., 0., 0.), 1)
            img = cv.putText(img, "REWARD: {0:0.2f}".format(self.last_reward), (70, 110), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0., 0., 0.), 1)
            img = cv.putText(img, "REWARD CUM: {0:0.2f}".format(self.reward_cum), (70, 130), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0., 0., 0.), 1)

        if self.done:
            img = cv.putText(img, "DONE", (270, 230), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0., 0., 1.), 3)
            img = cv.putText(img, "PRESS ENTER TO RESTART", (50, 270), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0., 0., 1.), 3)

        return img
        #cv.imshow('board', img)