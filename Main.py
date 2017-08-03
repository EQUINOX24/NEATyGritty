import NEATyGritty as NEAT
import Task_PB as Tsk
#import matplotlib.pyplot as plt
import pylab as plt

#import matplotlib
#from copy import deepcopy, copy

#matplotlib.rcParams['backend'] = "Qt4Agg"

NEAT.Inputs = Tsk.InOut['in']
NEAT.Outputs = Tsk.InOut['out']
NEAT.OutFunc = Tsk.InOut['activations']

NEAT.pool.firstGeneration()

plt.ion()
#plt.plot([0,3,2])
#plt.show()

def Loop():
    x = {}
    y = {}
    for g in range(1,501): # generation
        Fitness = []
        Tsk.Randomize()
        for specie in NEAT.pool.species:
            for network in specie.networks:
                Fitness.append(Tsk.Evaluator(network))
        plt.pause(0.0001)
        for specie in NEAT.pool.species:
            if specie.ID in x:
                x[specie.ID].append(g)
                y[specie.ID].append(specie.topFitness)
            else:
                x[specie.ID] = [g]
                y[specie.ID] = [specie.topFitness]
        print('g: ' + str(NEAT.pool.generation) + ', nodes: ' + str(len(NEAT.pool.species[0].networks[0].nodes)) \
        + ', max fitness: ' + ('%.2f' % (100*max(Fitness))) + '%, #species: ' + str(len(NEAT.pool.species)))
        NEAT.pool.newGeneration(g)
        if g%10 == 0:
            plt.clf()
            for ID in x:
                plt.plot(x[ID], y[ID])#, linewidth=2)
            print('g%10 == 0')
            plt.draw()
    plt.ioff()
    plt.show()
Loop()
