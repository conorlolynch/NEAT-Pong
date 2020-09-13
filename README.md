# NEAT-Pong

Using the NEAT-Python module to train a paddle to play pong

## Getting Started

Just run the pong.py file and watch as the AI begins training itself.

### Prerequisites

The only two modules that need to be installed are:

```
NEAT-Python (version 0.92 in this project)
Pygame (version 1.9.6 in this project)
```

## Demonstration

![alt text](Images/pong-gif.gif)

By generation 11 there seems to be a AI capable of playing pong.

## Explanation

Generation 0 is filled with a population of 50 paddles that do not know how to play pong. Paddles which collide with the ball are considered to be successful and are rewarded with a better fitness score once that generation is over. Paddles that fail to bounce the ball back are given a poor fitness score. At the end of a generation the most successful paddles with the highest fitness score are copied into the next generation of paddles and are crossed over with other sucessful paddles as well as having mutations performed on them to see if more successful hybrids can be created.  

After a number of generations the fitness of the entire population continues to improve to the point where the paddles can play pong well and can become unbeatable.
