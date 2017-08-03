from random import random, shuffle
from math import sqrt
from NEATyGritty import Act

InOut = { 'in': 2, 'out': 1, 'activations' : [Act.LOGISTIC]}

Input = [(0, 0), (1, 0), (0, 1), (1, 1)]
Output = [0, 1, 1, 0]
varKeep = { 'idx' : 32*list(range(0,len(Input))) }

def Randomize():
    shuffle(varKeep['idx'], lambda: random())

def Evaluator(NN):
    # fitness function:
    error = 0.0
    for i in range(0,len(varKeep['idx'])):
        error += abs(Output[varKeep['idx'][i]] - NN.Evaluate(Input[varKeep['idx'][i]])[0])
    NN.fitness = 1 - error/len(varKeep['idx'])
    return NN.fitness
