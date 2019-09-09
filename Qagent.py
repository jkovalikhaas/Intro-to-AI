from ple import PLE
import frogger_new
import numpy as np
import random
import sys
from pygame.constants import K_w,K_a,K_F15


# creates a unique value to differentiate states
class HashState:
    # seed is the value used to later 'hash'
    def __init__(self, seed):
        # keeps separate car/river 'hashes' closer to seed
        # also keeps starting state at '0'
        self.car_base = 0
        self.river_base = 0
        self.seed = self.set_seed(seed)

    # gets new value for state, different if frog is on street or river
    def add_table(self, value, cars):
        if cars:
            return self.set_car_state(value) % self.seed
        else:
            return self.set_river_state(value) % self.seed

    # gets the original seed value, all values in state added together
    def set_seed(self, state):
        for i in state['cars']:
            self.car_base += i.left + i.top + i.width + i.height
        for i in state['rivers']:
            self.river_base += i.left + i.top + i.width + i.height
        for i in state['homeR']:
            self.river_base += i.left + i.top + i.width + i.height
        total = state['frog_x'] + state['frog_y'] + state['rect_w'] + state['rect_h'] + self.car_base + self.river_base
        # returns the next prime in order to ensure unique values
        return self.next_prime(int(total))

    def set_river_state(self, state):
        total = state['frog_x'] + state['frog_y'] + state['rect_w'] + state['rect_h']
        for i in state['rivers']:
            total += i.left + i.top + i.width + i.height
        for i in state['homes']:
            # don't include flys or crocs
            if i == 0.66:
                total += i
        for i in state['homeR']:
            total += i.left + i.top + i.width + i.height
        total += self.car_base
        return int(total)

    def set_car_state(self, state):
        total = state['frog_x'] + state['frog_y'] + state['rect_w'] + state['rect_h']
        for i in state['cars']:
            total += i.left + i.top + i.width + i.height
        total += self.river_base
        return int(total)

    # finds the next prime
    def next_prime(self, value):
        value += 1
        if value % 2 == 0:
            value += 1
        while True:
            for i in range(2, int(value / 2)):
                if value % i == 0:
                    break
            else:
                return value
            value += 1


class NaiveAgent:
    def __init__(self, actions):
        self.actions = actions
        self.step = 0
        self.NOOP = K_F15

    # takes the state and also the array created from the imported file
    def pickAction(self, imprt, obs):
        # reward values:
        #   death: -1.0, midway: 0.1, end: 1.0, else: 0.0
        midpoint = 261.0

        num_actions = 5        # (x) actions: up,right,down,left,stay(NOOP)
        num_states = h.seed    # (y) arbitrary large number to account for various states
        # empty Q table
        q = np.zeros((num_states, num_actions))
        # if imported file exists, set q accordingly
        if imprt is not None:
            q = imprt

        iterations = 100000     # iterations before quitting game
        moves_per = 200         # soft caps on moves able to be made in one death

        alpha = 0.1     # learning rate
        gamma = 0.8     # discount rate

        for i in range(iterations):
            # print('i', i)
            # frogs = 0

            p.reset_game()  # reset game at start of iteration
            state = obs
            # boolean if frog has reached midpoint
            midpoint = state['frog_y'] > midpoint
            s = h.add_table(state, midpoint)    # gets states 'value'

            for j in range(moves_per):
                # random choice
                midpoint = state['frog_y'] >= midpoint    # if true, frog is before midpoint
                # temp array to hold when choosing random direction
                temp = [0, 1, 2, 3, 4]
                if midpoint:
                    # before midpoint, ignore down and stay actions
                    q[s, 2] = -1000.0
                    q[s, 4] = -1000.0
                    temp = [0, 1, 3]

                a = np.argmax(q[s, :])
                # gets random action
                if np.random.randint(0, 50) == 0:
                    a = random.choice(temp)

                # sets reward
                r = p.act(self.actions[a])
                if r == 0.1:
                    r = 1.5
                elif r == 1.0:
                    r = 5.0
                elif r == -1.0:
                    r = -5.0

                # for home in state['homes']:
                #     if home == 0.66:
                #         frogs += 1
                # if frogs >= 3:
                #     print(frogs)
                #     break
                # else:
                #     frogs = 0

                midpoint = state['frog_y'] > midpoint
                # gets new state and its value
                new_state = game.getGameState()
                new_s = h.add_table(new_state, midpoint)
                # sets q's new learned value
                q[s, a] = (1 - alpha) * q[s, a] + alpha * (r + gamma * np.max(q[new_s, :]))

                state = new_state
                s = new_s

                if p.game_over():
                    # counts frogs who made it home
                    break

            # saves output every 1000 iterations, or if frog has made significant progress
            # if frogs >= 2 or i % 1000 == 0 and i != 0:
            #     print(state)
            #     if frogs >= 2:
            #         np.savetxt('finished_' + str(frogs) + '.txt', q)
            #         if frogs == 5:
            #             break
            #     else:
            #         np.savetxt('test_done.txt', q)

        return self.NOOP
        # Uncomment the following line to get random actions
        # return self.actions[np.random.randint(0,len(self.actions))]

    # creates an array from file
    def file_to_array(self, contents, height):
        a = np.zeros((height, 5))
        y = 0
        for line in contents:
            line = line.split(' ')
            x = 0
            for i in line:
                i = float(i.strip('\n'))
                a[y, x] = i
                x += 1
            y += 1
        return a


game = frogger_new.Frogger()
fps = 30
p = PLE(game, fps=fps, force_fps=False)
agent = NaiveAgent(p.getActionSet())
reward = 0.0
h = HashState(game.getGameState())  # sets up hash value
# reads in arguments/file names
f = sys.argv[1]
o = open(f, 'r')
array = agent.file_to_array(o, h.seed)
# if no third argument was given, use '2'
arg = 2
if len(sys.argv) == 3:
    arg = int(sys.argv[2])

if arg == 1:
    # if just using table contents, not learning
    while True:
        if p.game_over():
            p.reset_game()

        obs = game.getGameState()
        mid = obs['frog_y'] > 261.0
        obs_value = h.add_table(obs, mid)
        action = p.act(agent.actions[np.argmax(array[obs_value])])
else:
    # if 0, starts table from scratch, otherwise resumes from file
    if arg == 0:
        array = None
    # runs learning
    obs = game.getGameState()
    action = agent.pickAction(array, obs)
