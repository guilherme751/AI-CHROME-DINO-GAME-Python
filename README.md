# AI-CHROME-DINO-GAME-Python

## Introdução
This repository presents an intelligent agent that plays the chrome dino game by Google. The implementation is made in Python programming language,
with the PyGame library. 

## Classifier
In order to classify the Dino's actions, I implemented a simple neural network that receives the state of the game, with informations such as 
the type of the next obstacle, the distance from the next obstascle, the height from the next obstacle and the speed of the game. It also receives
a weights set. The network has 4 inputs and 1 output, with 1 hidden layer with 4 neurons. So, based on the inputs, it calculated the output that will decide
whether the Dino must squat or jump.

## Metaheuristic
I used a genetic algorithm to select the best sets of weights. It does this by selecting the best solutions in each iteration. The top best ones will go 
straight to the next generation, as they will be part of the elite set. The others will have a chance to be selected based on their perfomance, and then they
will have a chance to go through a crossover and a mutation. The perfomance of a solutions (a set of weights to the neural network) is determined by the average
of three executions of the game.

## Training and Testing
For training the Dino, it's necessary to run the metaheuristics for over 1000 iterations. You can see the progress of the Dino in the terminal. For testing,
it will run the game 30 times and the average minus the standard deviation is the final score of the agent. 

## Collaborations
Any collaborations are welcome. It might be good to create another implementation for the crossover method and/or use another metaheuristic. The best score I got
was 1697.82 as the average of the 30 executions.

## Running
```
git clone git@github.com:guilherme751/trab2_IA.git
cd AI-CHROME-DINO-GAME-Python
python3 ./main.py {args} # Run with flag -h to check the arguments, otherwise it will run on default.
```





