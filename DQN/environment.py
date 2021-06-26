from collections import namedtuple
import pygame
import numpy as np

Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20
WIDTH = 200
HEIGHT = 200

# COLOR CODES
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREY = (224,224,224)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)


class Drone:
    
    def __init__(self, size, color):
        
        self.size = size
        self.color = color

    def place_drone(self, screen_width, screen_height):
        
        x = self.size + 20
        y = screen_height - self.size - 20

        return x, y

    def move(self, x, y, move_distace, choice):
        
        if choice == 0:  # left
            x -= move_distace
        if choice == 1:  # right
            x += move_distace
        if choice == 2:  # up
            y += move_distace
        if choice == 3:  # down
            y -= move_distace

        return x, y

class Man:
    
    def __init__(self, size, color):
        
        self.size = size
        self.color = color

    def place_man(self, screen_width, screen_height):
        
        x = screen_width - self.size 
        y = self.size + 20
        return x, y

class Environment:

    def __init__(self, w=WIDTH, h=HEIGHT):
        self.drone = Drone(BLOCK_SIZE, BLUE1)
        self.man = Man(BLOCK_SIZE, RED)
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))  # init display
        pygame.display.set_caption('SAR')
        self.iteration = 0
        self.constant = 100
        self.alpha = 0.01
        self.reset()

    def reset(self):
        
        self.drone_x, self.drone_y = self.drone.place_drone(WIDTH, HEIGHT)
        self.man_x, self.man_y = self.man.place_man(WIDTH, HEIGHT)
        self.iteration = 0

        return np.array([self.drone_x, self.drone_y])

    # Check if drone hits the boundary
    def is_drone_outside(self):
        
        if self.drone_x > self.w - BLOCK_SIZE or self.drone_x < 0 or self.drone_y > self.h - BLOCK_SIZE or self.drone_y < 0:
            return True
        return False
    
    # Check if man hits the boundary
    def is_man_outside(self):
        
        if self.man_x > self.w - BLOCK_SIZE or self.man_x < 0 or self.man_y > self.h - BLOCK_SIZE or self.man_y < 0:
            return True
        return False

    def get_reward(self):
        """returns reward and done"""
        if self.is_drone_outside() or self.is_man_outside():
            return -300, True
        elif self.drone_x == self.man_x and self.drone_y == self.man_y:
            return 500, True
        else:
            return -1, False

    def step(self, action):
        
        self.iteration += 1
        
        if self.iteration % 3 == 0:
            self.man_x, self.man_y = self.drone.move(
                self.man_x, self.man_y, 20, 0)
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.drone_x, self.drone_y = self.drone.move(
            self.drone_x, self.drone_y, BLOCK_SIZE, action)

        self.reward, self.done = self.get_reward()

        return np.array([self.drone_x, self.drone_y]), self.reward, self.done

    def render(self):
        
        self.display.fill(GREY)

        pygame.draw.rect(self.display, self.man.color, pygame.Rect(
            self.man_x, self.man_y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, self.drone.color, pygame.Rect(
            self.drone_x, self.drone_y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.display.flip()

############## COMMENT CODE ###############

# def place_man(self, drone_x, drone_y, screen_width, screen_height):

    #     y_positions = np.arange(0, 181, 20)
    #     y_random_position = np.random.choice(y_positions)

    #     x = screen_width - self.size - 0
    #     y = y_random_position

    #     # x = screen_width - self.size - 0
    #     # y = 0 + self.size + 20

    #     return x, y

# RESET 

        # rand_pos = np.random.randint(1, 3)
        # if rand_pos == 1:
        #     self.man_x, self.man_y = self.man.place_man(
        #         self.drone_x, self.drone_y, WIDTH, HEIGHT)
        # elif rand_pos == 2:
        #     self.man_x, self.man_y = self.man.place_man_2(
        #         self.drone_x, self.drone_y, WIDTH, HEIGHT)
    
# REWARD 

# elif (self.drone_y == self.man_y) and (self.drone_x != self.man_x):
        #     distance = self.relative_distance()
        #     reward_curr = round(
        #         math.log(self.constant/(self.alpha*distance)), 2)
        #     return reward_curr, False

        # elif (self.drone_y == self.man_y + 20) or (self.drone_y == self.man_y - 20):
        #     return 5, False
# def relative_distance(self):
#         return abs(self.drone_x - self.man_x)

#def move_man(self, man_x, man_y, pixel_per_step, direction):
        # if direction == 'left':
        #     man_x -= pixel_per_step
        # if direction == 'right':
        #     man_x += pixel_per_step
        # if direction == 'down':
        #     man_y += pixel_per_step
        # if direction == 'up':
        #     man_y -= pixel_per_step

        # return man_x, man_y