import pygame
import random
from enum import Enum
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Constants 
BLOCK_SIZE = 20
INITIAL_SPEED = 30
MAX_SPEED = 100
SPEED_UP_EVERY = 5  # apples
SPEED_STEP = 5

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# Setup 
pygame.init()
font = pygame.font.SysFont('arial', 25)

# Helpers
class Direction(Enum):
    UP    = 1
    DOWN  = 2
    LEFT  = 3
    RIGHT = 4

Point = namedtuple('Point', 'x y')

WHITE  = (255,255,255)
RED    = (200,0,0)
GREEN1 = (124,252,0)
GREEN2 = (50,205,50)
BLACK  = (0,0,0)


# Snake Game Environment 
class SnakeGameAI:
    def __init__(self, width=640, height=480):
        self.w, self.h = width, height
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake with DQN')
        self.clock = pygame.time.Clock()

        self.high_score = self._load_high_score()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        mid_x = self.w // 2
        mid_y = self.h // 2
        self.head = Point(mid_x, mid_y)
        self.snake = [                                                 
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2*BLOCK_SIZE, self.head.y)
        ]
        self.snake_set = set(self.snake)

        self.blocks = []
        self.blocks_set = set()

        self.score = 0
        self.frame_iteration = 0
        self.speed = INITIAL_SPEED

        self._place_food()

    def _load_high_score(self, fname='high_score.txt'):
        try:
            with open(fname) as f:
                return int(f.read())
        except:
            return 0

    def _save_high_score(self, fname='high_score.txt'):
        if self.score > self.high_score:
            with open(fname, 'w') as f:
                f.write(str(self.score))
            self.high_score = self.score

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        pt = Point(x,y)
        if pt in self.snake_set or pt in self.blocks_set:
            return self._place_food()
        self.food = pt

    def _add_block(self):
        free = []
        for x in range(0, self.w, BLOCK_SIZE):
            for y in range(0, self.h, BLOCK_SIZE):
                p = Point(x,y)
                if p not in self.snake_set and p not in self.blocks_set and p != self.food:
                    free.append(p)
        if free:
            b = random.choice(free)
            self.blocks.append(b)
            self.blocks_set.add(b)

    def play_step(self, action):
        self.frame_iteration += 1
        # 1) handle quit & pause
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_p]:
            paused = True
            while paused:
                for ev in pygame.event.get():
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_p:
                        paused = False
                self.clock.tick(5)

        # 2) move
        self._move(action)
        self.snake.insert(0, self.head)
        self.snake_set.add(self.head)

        # 3) check collision
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            self._save_high_score()
            return reward, game_over, self.score

        # 4) eat food
        if self.head == self.food:
            self.score += 1
            reward = 10
            # speed up every few apples
            if self.score % SPEED_UP_EVERY == 0:
                self.speed = min(self.speed + SPEED_STEP, MAX_SPEED)
            # add a block and place new food
            self._add_block()
            self._place_food()
        else:
            # move forward: drop tail
            tail = self.snake.pop()
            self.snake_set.remove(tail)

        # 5) update UI and timing
        self._update_ui()
        self.clock.tick(self.speed)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # wall
        if pt.x < 0 or pt.x >= self.w or pt.y < 0 or pt.y >= self.h:
            return True
        # self
        if pt in list(self.snake)[1:]:
            return True
        # blocks
        if pt in self.blocks_set:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        # draw snake
        for p in self.snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(p.x, p.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(p.x+4, p.y+4, 12, 12))
        # draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # draw blocks
        for b in self.blocks:
            pygame.draw.rect(self.display, WHITE, pygame.Rect(b.x, b.y, BLOCK_SIZE, BLOCK_SIZE))
        # draw scores
        score_txt = font.render(f"Score: {self.score}  High Score: {self.high_score}", True, WHITE)
        self.display.blit(score_txt, (10, 10))
        pygame.display.flip()

    def _move(self, action):
        # action: [straight, right, left]
        clock = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock.index(self.direction)
        if np.array_equal(action, [1,0,0]):
            new_dir = clock[idx]
        elif np.array_equal(action, [0,1,0]):
            new_dir = clock[(idx+1)%4]
        else:
            new_dir = clock[(idx-1)%4]
        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        else:  # UP
            y -= BLOCK_SIZE
        self.head = Point(x, y)


# Deep Q‑Network Components
class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state  = torch.tensor(np.array(state), dtype=torch.float)
        next_s = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            # batch of 1
            state  = state.unsqueeze(0)
            next_s = next_s.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # 1) predicted Q values with current state
        pred = self.model(state)

        # 2) target = reward + gamma * max(next_pred) if not done
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_s[idx]))
            target[idx][ torch.argmax(action[idx]).item() ] = Q_new

        # 3) backprop
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


class Agent:
    def __init__(self):
        self.n_games  = 0
        self.epsilon  = 0  # randomness
        self.gamma    = 0.9
        self.memory   = deque(maxlen=MAX_MEMORY)
        self.model    = LinearQNet(11, 256, 3)
        self.trainer  = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game: SnakeGameAI):
        head = game.head
        # danger straight, right, left
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # move direction
            dir_l, dir_r, dir_u, dir_d,

            # food location
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y   # food down
        ]
        return np.array(state, dtype=int)

    def remember(self, *args):
        self.memory.append(args)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini = random.sample(self.memory, BATCH_SIZE)
        else:
            mini = self.memory
        states, actions, rewards, next_states, dones = zip(*mini)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, *args):
        self.trainer.train_step(*args)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final = [0,0,0]
            final[move] = 1
            return final
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            preds = self.model(state0)
            move = torch.argmax(preds).item()
            final = [0,0,0]
            final[move] = 1
            return final


# Training 
def train():
    plot_scores = []
    total_score = 0
    agent = Agent()
    game  = SnakeGameAI()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get action
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            total_score += score
            mean_score = total_score / agent.n_games
            print(f'Game {agent.n_games}  Score {score}  Mean {mean_score:.2f}')

if __name__ == '__main__':
    train()
