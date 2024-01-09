import game
import cv2 as cv
from agent import SnakeAgent
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

class Mode(Enum):
    Train = 1
    Play = 2

def display_info():
    img = np.ones((300, 300))
    img = cv.putText(img, 'Max Score: {0}'.format(board.max_score), (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0., 0., 0.), 1)
    img = cv.putText(img, 'Train Score: {0:0.4f}'.format(np.average(scores)), (30, 45), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0., 0., 0.), 1)
    img = cv.putText(img, 'Test Score: {0:0.4f}'.format(np.average(test_scores)), (30, 60), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0., 0., 0.), 1)
    img = cv.putText(img, 'Epsilon: {0:0.3f}'.format(epsilon), (30, 75), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0., 0., 0.), 1)
    img = cv.putText(img, 'Episode: {0}/{1}'.format(episode, episodes), (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0., 0., 0.), 1)
    for i in range(len(agent.actions)):
        img = cv.putText(img, 'action[{0}]: {1}'.format(i, agent.actions[i]), (30, 105+i*15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0., 0., 0.), 1)

    cv.imshow('Info', img)
    cv.waitKey(1)

def mavg(data, period = 20):
    mavg = []
    for i in range(len(data)):
        if i < period:
            mavg.append(np.average(data[0:i]))
        else:
            mavg.append(np.average(data[i-period:i]))
    return mavg

def plot_score(scores, test_scores):
    plt.figure(figsize=(10, 8))
    #plt.plot(scores, label='score')
    plt.plot(mavg(scores, mavg_periods), label='Train')
    plt.plot(mavg(test_scores, mavg_periods), label='Test')
    plt.xlabel("episode")
    plt.ylabel("score")
    plt.legend()
    plt.title('Score')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.xlim(0, len(scores)-1)
    plt.ylim(0, ymax)
    plt.minorticks_on()
    plt.grid(which='both')
    #plt.show()
    plt.savefig('accs.png')
    plt.close()

def plot_reward(rewards, test_rewards):
    plt.figure(figsize=(10, 8))
    #plt.plot(scores, label='score')
    plt.plot(mavg(rewards, mavg_periods), label='Train')
    plt.plot(mavg(test_rewards, mavg_periods), label='Test')
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.legend()
    plt.title('Reward')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.xlim(0, len(scores)-1)
    #plt.ylim(0, ymax)
    plt.minorticks_on()
    plt.grid(which='both')
    #plt.show()
    plt.savefig('rewards.png')
    plt.close()

def test(periods):
    scores = []
    rewards = []
    for period in range(periods):
        board.reset()
        while not board.done:
            with torch.no_grad():
                direction = agent.act(board.state.reshape((1, board.state.shape[0], board.state.shape[1])), epsilon_test)
            board.step_direction(direction)
            if render:
                img = board.render(agent.actions)
                cv.imshow('Test', img)
                cv.waitKey(1)
            pass
        scores.append(board.score)
        rewards.append(board.reward_cum)
        #print('Test {0}/{1} score {2:0.4f}'.format(episode, episodes, np.average(scores)))
    return np.average(scores), np.average(rewards)

mode = Mode.Train
game.DEBUG = True
render = True

episodes = 5000
epsilon = 1.
batch_size = 64
gamma = 0.9
buffer_size = 20000
update_model_steps = 10#batch_size * 2
update_target_model_episodes = 50
epsilon_start = 0.4
epsilon_end = 0.
save_model_episodes = 100
filename = 'model.sd'
mavg_periods = 1000
epsilon_test = 0.01
store_zero_score_prop = 0.5

board = game.SnakeBoard(size=(10,10))
board.render()

agent = SnakeAgent(buffer_size, gamma)
if os.path.isfile(filename):
    with open(filename, 'rb') as f:
        agent.model.load_state_dict(torch.load(f))
    with open(filename, 'rb') as f:
        agent.target_model.load_state_dict(torch.load(f))
scores = []
test_scores = []
rewards = []
test_rewards = []
episode = 1
step = 1
if mode == Mode.Train:
    while episode <= episodes:
        board.reset()
        epsilon = epsilon_start - (episode / episodes) * (epsilon_start - epsilon_end)
        last_state = board.state
        while not board.done:
            state = board.state.reshape((1, board.state.shape[0], board.state.shape[1]))
            action = agent.act(state, epsilon)

            reward = board.step_direction(action)
            next_state = board.state.reshape((1, board.state.shape[0], board.state.shape[1]))

            if reward != game.REWARD_DEFAULT or np.random.random() <= store_zero_score_prop:
                agent.erm.append(state, action, reward, next_state, board.done, store_zero_reward_prop=1.)

            if step%update_model_steps == 0:
                agent.train(batch_size)

            if render:
                cv.imshow('Train', board.render(agent.actions))
                cv.waitKey(1)
                display_info()
            step += 1

        scores.append(board.score)
        rewards.append(board.reward_cum)

        if episode%update_target_model_episodes == 0:
            agent.update_target_model()

        test_score, test_reward = test(1)
        test_scores.append(test_score)
        test_rewards.append(test_reward)
        plot_score(scores, test_scores)
        plot_reward(rewards, test_rewards)

        test_score = 0 if len(test_scores)==0 else test_scores[-1]
        print('Epsiode {0}/{1} eps {2:0.4f} score {3:0.3f} test_score {4:0.3f} max_score {5} reward {6:0.2f} test_reward {7:0.2f}'.format(episode, episodes, epsilon, scores[-1], test_score, board.max_score, rewards[-1], test_rewards[-1]))

        if episode%save_model_episodes == 0:
            with open(filename, 'wb') as f:
                torch.save(agent.model.state_dict(), f)

        episode += 1
if mode == Mode.Play:
    key = -1
    while True:
        # Left: 2424832 Up: 2490368 Right: 2555904 Down: 2621440
        if board.done:    # return
            if key == 13:
                board.reset()
        else:
            #if key == 2:                # left arrow
            #    board.step(-1)
            #elif key == 3:              # right arrow
            #    board.step(1)
            #else:
            if game.DEBUG:
                #board.step_direction(key)
                if key == 0:        # arrow up
                    board.step_direction(0)
                if key == 1:        # arrow down
                    board.step_direction(2)
                if key == 2:        # arrow left
                    board.step_direction(3)
                if key == 3:        # arrow right
                    board.step_direction(1)
            else:
                board.step(0)
        cv.imshow('Snake', board.render())

        if game.DEBUG:
            key = cv.waitKey(-1)
        else:
            key = cv.waitKey(200)
        #time.sleep(0.4)