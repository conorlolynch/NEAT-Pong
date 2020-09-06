import pygame as pg
import random
import neat, neat.config
import os

pg.init()
screen = pg.display.set_mode((800,600))
screen_rect = screen.get_rect()
clock = pg.time.Clock()

myfont = pg.font.SysFont('freesansbold.ttf', 40)

# Getting our path to our configuration file for NEAT module
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "config-feedforward.txt")
gen = -1

class Ball:
    def __init__(self, screen_rect, size):
        self.screen_rect = screen_rect
        self.height, self.width = size
        self.image = pg.Surface(size).convert()
        self.image.fill((255,0,0))
        self.rect = self.image.get_rect()
        self.speed = 10
        self.velocityChange = False
        self.set_ball()

    def get_random_float(self):
        '''get float for velocity of ball on starting direction'''
        while True:
            num = random.uniform(-1.0, 1.0)
            if num > -.5 and num < .5: #restrict ball direction to avoid infinity bounce
                continue
            else:
                return num

    def set_ball(self):
        '''get random starting direction and set ball to center screen'''
        x = abs(self.get_random_float())
        y = self.get_random_float()
        self.vel = [x, y]
        self.rect.center = self.screen_rect.center
        self.true_pos = list(self.rect.center)


    def collide_walls(self):
        # Check if we hit the top or bottom of screen
        if (self.rect.y < 0) or (self.rect.y > self.screen_rect.bottom - self.height):
            self.vel[1] *= -1;

        if (self.rect.x > self.screen_rect.right- self.height):
            self.vel[0] *= -1;


    def checkCollideZone(self, paddle):
        # Check if we hit the left wall so the game and variables will restart
        if (self.rect.x < 0):
            # We need to restart the game as well as
            self.set_ball()
            paddle.ballHits = 0
            ''' Probably need to delete this paddle '''

    def collide_paddle(self, paddle, velocityChange):
        if self.rect.colliderect(paddle.rect):
            self.vel[0] = abs(self.vel[0]);
            paddle.ballHits += 1
            return True

        return False


    def move(self):
        self.true_pos[0] += self.vel[0] * self.speed
        self.true_pos[1] += self.vel[1] * self.speed
        self.rect.center = self.true_pos


    def update(self, paddle):
        self.collide_walls(paddle)
        self.collide_paddle(paddle)
        self.move()

    def render(self, screen):
        screen.blit(self.image, self.rect)

class Paddle:
    def __init__(self, screen_rect, size, trainingNeuralNetwork, color):
        self.screen_rect = screen_rect
        self.width, self.height = size
        self.image = pg.Surface(size).convert()
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x += 25 #spacer from wall
        self.speed = 5      # Usual speed = 5
        self.trainingNeuralNetwork = trainingNeuralNetwork
        self.ballHits = 0



    def move(self, x, y):
        self.rect[0] += x * self.speed
        self.rect[1] += y * self.speed

        # Check to make sure not overlapping edge of screen
        if (self.rect[1] < 0):
            self.rect[1] = 0

        if (self.rect[1] > self.screen_rect.bottom - self.height):
            self.rect[1] = self.screen_rect.bottom - self.height

    def moveUp(self):
        self.move(0, -1)

    def moveDown(self):
        self.move(0, 1)

    def update(self, keys):
        if (self.trainingNeuralNetwork):
            pass
        else:
            self.rect.clamp_ip(self.screen_rect)
            if keys[pg.K_UP] or keys[pg.K_w]:
                self.move(0, -1)
            if keys[pg.K_DOWN] or keys[pg.K_s]:
                self.move(0, 1)


    def render(self, screen):
        screen.blit(self.image, self.rect)


def main(genomes, config):

    # Need to keep track of the neural networks that control each paddle
    nets = []
    ge = []
    players = []
    to_remove = []

    # Setup neural network for that genome, create paddle object for that network
    for _,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        players.append(Paddle(screen_rect, (25,100), trainingNeuralNetwork = True, color=(random.randint(0,255), random.randint(0,255), random.randint(0,255))))
        g.fitness = 0
        ge.append(g)


    global gen
    gen += 1

    done = False
    highestScore = 0
    velocityChange = False

    ball = Ball(screen_rect, (25,25))


    while not done and len(players) > 0:
        to_remove = []
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True

        screen.fill((0,204,204))

        # Need to move all the ai paddles
        for x, player in enumerate(players):
            # Pass some values to the neural network for this paddle and get its output value
            output = nets[x].activate((player.rect[1], abs(player.rect[1] - ball.rect[1]), abs((player.rect[1] + player.height) - ball.rect[1])))

            # Now look at this output and see if its greater than 0.5
            if (output[0] > 0.5):
                player.moveUp()
            elif (output[1] < -0.5):
                player.moveDown()
            else:
                pass

        # Do all the ball movement stuff
        ball.move()
        ball.collide_walls()

        # Check for collisions with each paddle and leftzone
        for x, paddle in enumerate(players):
            if ((ball.rect[0] < (paddle.rect[0] + paddle.width))):
                if (ball.collide_paddle(paddle, velocityChange)):
                    # We want to keep this paddle and give it better fitness
                    velocityChange = True
                    ge[x].fitness += 5
                else:
                    # We want to get rid of this paddle and give it a worse fitness score
                    # Every time a paddle lets the ball through it loses a fitness score
                    # This encourages the network to attempt to hit the ball
                    to_remove.append(x)


            ball.checkCollideZone(paddle)


        # Remove all the paddles that didnt hit the ball back
        for x in sorted(to_remove,reverse=True):
            ge[x].fitness -= 1
            players.pop(x)
            nets.pop(x)
            ge.pop(x)


        # Make sure velocity of the ball can change for next tick
        velcoityChange = False

        # Draw all the ai paddles and get their highest score yet
        for paddle in players:
            paddle.render(screen)
            if paddle.ballHits > highestScore:
                highestScore = paddle.ballHits

        ball.render(screen)

        # Get the current highest return score
        high_score_box = myfont.render(("High score: "+str(highestScore)), False, (100, 100, 100))
        generation_box = myfont.render(("Gen: "+str(gen)), False, (100, 100, 100))
        num_alive_box  = myfont.render(("Alive: "+str(len(players))), False, (100, 100, 100))
        screen.blit(high_score_box,(100, 20))
        screen.blit(generation_box,(350, 20))
        screen.blit(num_alive_box,(500, 20))

        clock.tick(60)
        pg.display.update()


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    # Next we create a population
    p = neat.Population(config)

    # Add stat reporters (give us useful output from generations)
    p.add_reporter(neat.StdOutReporter())
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Last step is to set up the fitness function
    # We call the main function 50 times an evaluate that
    # Needs to take all our genomes and get their fitness of all the paddles
    winner = p.run(main, 50)


run(config_path)
