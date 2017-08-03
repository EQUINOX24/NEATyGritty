from math import  exp, tanh, ceil, floor, sqrt
from statistics import median
from copy import deepcopy, copy
from random import random, randint, gauss

Inputs = 2
Outputs = 1
OutFunc = []

DeltaDisjoint = 2.0
DeltaWeights = 0.4
SimilarityThreshold = 1.3

StaleSpecie = 15
PerturbChance = 0.90
PointReorderChance = 0.8

CrossoverChance = 0.75

WeightMutateChance = 0.25
LinkMutationChance = 1.8
NodeInsertionChance = 0.08
FuncMutationChance = 0.15
ReorderNodesChance = 0.25
ConnectToSelfChance = 0.15
EnableMutationChance = 0.2
DisableMutationChance = 0.4
StepSTDEV = 0.15

MaxValue = 12.0

# ========================================================================
# activation function parameters
param_1, param_2, param_3 = 2, 3, 1
def Logistic(x):
    # this will not cause domain errors unlike 1/(1+exp(-x))
    return (tanh(param_1*x) + 1)/2

def Gauss(x):
    return exp(-param_2*x**2)

def Tanh(x):
    return tanh(param_3*x)

ActFuncs = [Tanh, Logistic, Gauss] #atan,

class Act():
    TANH = 0
    LOGISTIC = 1
    GAUSS = 2

class NodeGene():
    ID = 0 # unique node number
    layer = 1 # {input: 0, hidden: 1, output: 2}
    #indexIO = 0 # position in input/output list. ignored in case hidden
    bias = 0.0 # bias of the node
    func = Act.TANH # activation function
    init_preact = 0.0 # initial preactivation value
    def __init__(self,ID):
        self.ID = ID

class LinkGene():
    From = 0
    To = 0
    weight = 1.0
    enabled = True
    innov = 0

class Neuron():
    value = 0.0
    incoming = None
    index = 0 # position index at nodes
    def __init__(self,index):
        self.incoming = []
        self.index = index

class Network():
    nodes = None # order of elements in this list determines the order of neuron activation
    links = None
    fitness = 0.0
    neurons = None
    globalRank = 0
    mutRates = None # mutation rates
    def __init__(self):
        self.nodes = []
        self.links = []
        self.fitness = 0.0
        self.network = []
        self.mutRates = {'weight': WeightMutateChance,
                         'link': LinkMutationChance,
                         'node': NodeInsertionChance,
                         'func': FuncMutationChance,
                         'reorder': ReorderNodesChance,
                         'self': ConnectToSelfChance,
                         'enable': EnableMutationChance,
                         'disable': DisableMutationChance,
                         'step': StepSTDEV}

    def GenerateNeurons(self):
        self.neurons = {}
        for n in range(0,len(self.nodes)):
            node = self.nodes[n]
            self.neurons[node.ID] = Neuron(n)
            self.neurons[node.ID].incoming = []
            self.neurons[node.ID].value = ActFuncs[node.func](node.init_preact)
        for link in self.links:
            if link.enabled:
                self.neurons[link.To].incoming.append(link)

    def _addDisabledLinks(self):
        for link in self.links:
            if not link.enabled:
                self.neurons[link.To].incoming.append(link)

    def Evaluate(self,inputs):
        outputs = [0.0]*Outputs
        for node in self.nodes:
            if node.layer == 0:
                self.neurons[node.ID].value = inputs[node.indexIO]
            else:
                Sum = node.bias
                for incoming in self.neurons[node.ID].incoming:
                    Sum += incoming.weight*self.neurons[incoming.From].value
                preactivation = Sum
                self.neurons[node.ID].value = ActFuncs[node.func](preactivation)
                if node.layer == 2:
                    outputs[node.indexIO] = self.neurons[node.ID].value
        return outputs

    def Mutate(self, linkChanceInit=False):
        # Before running this self.nodes should be up to date.
        for mutation in self.mutRates:
            if randint(0,1):
                self.mutRates[mutation] *= 0.95
            else:
                self.mutRates[mutation] *= 1.05263

        self.fixedTopologyMutation()

        p = self.mutRates['enable']
        while p > 0:
            if random() < p:
                self._toggleLink(enable=True)
            p -= 1

        p = self.mutRates['disable']
        while p > 0:
            if random() < p:
                self._toggleLink(enable=False)
            p -= 1

        self._addDisabledLinks()

        p = self.mutRates['self']
        while p > 0:
            if random() < p:
                self._selfLink()
            p -= 1

        p = 2.0 if linkChanceInit else self.mutRates['link']
        while p > 0:
            if random() < p:
                self._addLink()
            p -= 1

        p = self.mutRates['reorder']
        while p > 0:
            if random() < p:
                self._alterNodeOrder()
            p -= 1

        p = self.mutRates['node']
        while p > 0:
            if random() < p:
                self._insertNode()
            p -= 1

        self.GenerateNeurons()

    def fixedTopologyMutation(self):
        self._mutateLinkWeights()
        self._mutateNodes()

    def _mutateLinkWeights(self):
        for link in self.links:
            if random() < self.mutRates['weight']:
                if random() < PerturbChance:
                    link.weight += gauss(0, self.mutRates['step'])
                else:
                    if randint(0,1):
                        link.weight += gauss(0, 1.0)
                    else:
                        link.weight = gauss(0, 1.0)

                    if link.weight > MaxValue:
                        link.weight = MaxValue
                    elif link.weight < -MaxValue:
                        link.weight = -MaxValue

    def _mutateNodes(self):
        for node in self.nodes:
            if node.layer > 0:
                if random() < self.mutRates['weight']:
                    if random() < PerturbChance:
                        node.bias += gauss(0, self.mutRates['step'])
                    else:
                        if randint(0,1):
                            node.bias += gauss(0, 1.0)
                        else:
                            node.bias = gauss(0, 1.0)

                    if node.bias > MaxValue:
                        node.bias = MaxValue
                    elif node.bias < -MaxValue:
                        node.bias = -MaxValue

                if random() < self.mutRates['weight']:
                    if random() < PerturbChance:
                        node.init_preact += gauss(0, self.mutRates['step'])
                    else:
                        if randint(0,1):
                            node.init_preact += gauss(0, 1.0)
                        else:
                            node.init_preact = gauss(0, 1.0)

                    if node.init_preact > MaxValue:
                        node.init_preact = MaxValue
                    elif node.init_preact < -MaxValue:
                        node.init_preact = -MaxValue

                if node.layer == 1:
                    if random() < self.mutRates['func']:
                        r = randint(0, len(ActFuncs) - 1)
                        node.func = r

    def _toggleLink(self, enable=True):
        candidates = []
        for link in self.links:
            if link.enabled != enable:
                candidates.append(link)
        L = len(candidates)
        if L == 0:
            return
        r = randint(0,L-1)
        candidates[r].enabled = enable

    def _selfLink(self):
        candidates = []
        for node in self.nodes:
            if node.layer != 0:
                include = True
                for link in self.neurons[node.ID].incoming:
                    if link.To == link.From:
                        include = False
                        break
                if include:
                    candidates.append(node)
        L = len(candidates)
        if L == 0:
            return
        r = randint(0,L-1)
        newLink = LinkGene()
        newLink.To = candidates[r].ID
        newLink.From = candidates[r].ID
        newLink.weight = gauss(0, 1.0)
        newLink.innov = pool.newLinkInnov()

        self.links.append(newLink)
        self.neurons[candidates[r].ID].incoming.append(newLink)

    def _addLink(self):
        nodes_To,input_nodes = [],[]
        for g in range(0,len(self.nodes)):
            if self.nodes[g].layer != 0:
                nodes_To.append(g)
            else:
                input_nodes.append(g)
        while True:
            if len(nodes_To) == 0:
                return
            r = randint(0, len(nodes_To) - 1)
            To = self.nodes[nodes_To[r]].ID
            del nodes_To[r]
            nodes_From = nodes_To + input_nodes
            for link in self.links:
                if link.To == To:
                    try:
                        nodes_From.remove(self.neurons[link.From].index)
                    except:
                        pass
            if len(nodes_From) == 0:
                input_nodes.append(self.neurons[To].index)
                continue
            r = randint(0, len(nodes_From) - 1)
            From = self.nodes[nodes_From[r]].ID

            newLink = LinkGene()
            newLink.To = To
            newLink.From = From
            newLink.weight = gauss(0, 1.0)
            newLink.innov = pool.newLinkInnov()

            self.links.append(newLink)
            self.neurons[To].incoming.append(newLink)
            return

    def _alterNodeOrder(self):
        L = len(self.nodes)
        if L > 1:
            if random() < PointReorderChance:
                r = randint(0,L-1)
                node = self.nodes.pop(r)
                r = randint(0,L-1)
                self.nodes.insert(r,node)
            else:
                r1 = randint(0,L-1)
                r2 = randint(0,L-1)
                if r2 >= r1:
                    r2 += 1
                if r1 > r2:
                    r1,r2 = r2,r1
                cutout = self.nodes[r1:r2]
                self.nodes = self.nodes[0:r1] + self.nodes[r2:]
                r = randint(0, len(self.nodes))
                self.nodes = self.nodes[0:r] + cutout + self.nodes[r:]

    def _insertNode(self):
        # Disables a synaptic link while inserting a new node between nodes it used to connect.
        links = []
        for g in range(0,len(self.links)):
            if self.links[g].enabled:
                links.append(g)
        if len(links) == 0:
            return
        r = randint(0, len(links) - 1)
        link = self.links[links[r]]

        for n in  range(0,len(self.nodes)):
            if self.nodes[n].ID == link.From:
                func = self.nodes[n].func
                break
        for n in  range(0,len(self.nodes)):
            if self.nodes[n].ID == link.To:
                break

        node = NodeGene(pool.newNodeID())
        node.func = func

        link1 = copy(link)
        link2 = copy(link)

        link1.innov = pool.newLinkInnov()
        link2.innov = pool.newLinkInnov()

        link1.To = node.ID
        link2.From = node.ID

        # The following parameters will minimize the impact of this mutation:
        if func == Act.TANH:
            link1.weight = 1.87
            link2.weight = 1.04
        elif func == Act.LOGISTIC:
            link1.weight = 6.0
            node.bias = 3.0
        #elif func == Act.SOFTPLUS:
        #    link1.weight = 2.5
        #    node.bias = -1.66
        elif func == Act.GAUSS:
            link1.weight = 1.86
            node.bias = -2.0
        node.weight = link1.weight

        # Insert new link into the genome:
        self.links.append(link1)
        self.links.append(link2)
        self.nodes.insert(n,node)
        # Disable the old synaptic link:
        link.enabled = False

# ========================================================================

# Specie is not grammatically correct singular form of species, it's used for convenience
class Specie():
    topFitness = -float('inf')
    topFitnessEver = -float('inf')
    staleness = 0
    networks = None
    averageScore = 0
    ID = 0
    def __init__(self):
        self.staleness = 0 # ?
        self.networks = []

class Pool():
    population = 300
    species = None
    generation = 0
    link_innov = 0
    node_ID = -1
    s = 0
    def __init__(self):
        self.species = []

    def newSpecieID(self):
        self.s += 1
        return self.s

    def newLinkInnov(self):
        self.link_innov += 1
        return self.link_innov

    def newNodeID(self):
        pool.node_ID += 1
        return pool.node_ID

    def basicNetwork(self):
        basicNet = Network()
        for n in range(0,Inputs):
            Input = NodeGene(self.newNodeID())
            Input.layer = 0
            Input.indexIO = n
            basicNet.nodes.append(Input)
        for n in range(0,Outputs):
            Output = NodeGene(self.newNodeID())
            Output.layer = 2
            Output.indexIO = n
            Output.func = OutFunc[n]
            basicNet.nodes.append(Output)
        basicNet.GenerateNeurons()
        return basicNet

    def firstGeneration(self):
        basicNet = self.basicNetwork()
        S = ceil(0.02*self.population)
        for n in range(0, S):
            self.species.append(Specie())
            self.species[n].ID = self.newSpecieID()
            self.species[n].networks.append(deepcopy(basicNet))
            self.species[n].networks[0].Mutate(linkChanceInit=True)
        N = S
        while N < self.population:
            r1 = randint(0, S - 1)
            r2 = randint(0, len(self.species[r1].networks) - 1)
            child = deepcopy(self.species[r1].networks[r2])
            child.fixedTopologyMutation()
            self.species[r1].networks.append(child)
            N += 1
        self.generation += 1

    def newGeneration(self, g):
        self.cullSpecies(keep=0.50)
        self.rankGlobally()
        self.removeStaleSpecies()
        self.rankGlobally()
        self.calculateAverageScore()
        self.removeWeakSpecies()
        children, childOf = [], []
        totalAverageScore = self.totalAverageScore()
        for s in range(0,len(self.species)):
            specie = self.species[s]
            breed = floor(self.population * specie.averageScore / totalAverageScore) - 1
            for i in range(0,breed):
                children.append(self.createChild(specie))
                childOf.append(s)
        self.cullSpecies()
        L = 0
        for specie in self.species:
            for network in specie.networks:
                L += 1
        while len(children) + L < self.population:
            r = randint(0, len(self.species) - 1)
            children.append(self.createChild(self.species[r]))
            childOf.append(r)

        parentSpecies = len(self.species)
        for c in range(0,len(children)):
            self.addToSpecie(children[c], childOf[c], parentSpecies)

        self.generation += 1

    def cullSpecies(self,keep=0.1):
        for specie in self.species:
            specie.networks.sort(key = lambda x: x.fitness, reverse=True)
            remaining = ceil(keep * len(specie.networks))
            specie.networks = specie.networks[0:remaining]

    def rankGlobally(self):
        ranked = []
        for specie in self.species:
            for network in specie.networks:
                ranked.append(network)
        ranked.sort(key = lambda x: x.fitness)
        for n in range(0, len(ranked)):
            ranked[n].globalRank = n

    def removeWeakSpecies(self):
        totalAverageScore = self.totalAverageScore()
        survived = []
        for specie in self.species:
            breed = floor(self.population * specie.averageScore / totalAverageScore)
            if breed >= 1:
                survived.append(specie)
        self.species = survived

    def removeStaleSpecies(self):
        topFitnesses = []
        for specie in self.species:
            specie.networks.sort(key = lambda x: x.fitness, reverse=True)
            if specie.networks[0].fitness > specie.topFitnessEver:
                specie.topFitnessEver = specie.networks[0].fitness
                specie.staleness = 0
            else:
                specie.staleness += 1
            specie.topFitness = specie.networks[0].fitness
            topFitnesses.append(specie.topFitness)
        medFit = median(topFitnesses)
        survived = []
        for specie in self.species:
            if specie.staleness < StaleSpecie or specie.topFitness >= medFit:
                survived.append(specie)
        self.species = survived

    def calculateAverageScore(self):
        for specie in self.species:
            total = 0
            for network in specie.networks:
                total += network.globalRank
            specie.averageScore = total / len(specie.networks)

    def totalAverageScore(self):
        total = 0
        for specie in self.species:
            total += specie.averageScore
        return total

    def createChild(self, specie):
        if len(specie.networks) > 1 and random() < CrossoverChance:
            r1 = randint(0, len(specie.networks) - 1)
            r2 = randint(0, len(specie.networks) - 2)
            if r2 == r1:
                r2 += 1
            child = self.crossover(specie.networks[r1], specie.networks[r2])
            child.GenerateNeurons()
        else:
            r = randint(0, len(specie.networks) - 1)
            child = deepcopy(specie.networks[r])
        child.Mutate()
        return child

    def addToSpecie(self, child, childOf, parentSpecies):
        specie = self.species[childOf]
        if self.sameSpecie(child, specie.networks[0]):
            specie.networks.append(child)
            return
        for s in range(parentSpecies, len(self.species)):
            if self.sameSpecie(child, self.species[s].networks[0]):
                self.species[s].networks.append(child)
                return
        newSpecie = Specie()
        newSpecie.ID = self.newSpecieID()
        newSpecie.networks.append(child)
        self.species.append(newSpecie)

    def crossover(self, n1, n2):
        # Make sure n1 is the higher fitness network:
        if n2.fitness > n1.fitness:
            n1,n2 = n2,n1

        child = Network()

        child.mutRates = copy(n1.mutRates)


        links = [None]*(self.link_innov + 1)
        for link2 in n2.links:
            links[link2.innov] = link2

        nodes = [None]*(self.node_ID + 1)
        for node2 in n2.nodes:
            nodes[node2.ID] = node2

        # Link gene crossover:
        for link1 in n1.links:
            link2 = links[link1.innov]
            if link2 != None and randint(0,1):
                if link2.enabled:
                    child.links.append(copy(link2))
                    continue
            child.links.append(copy(link1))

        # Node gene crossover:
        for node1 in n1.nodes:
            node2 = nodes[node1.ID]
            if node2 != None and randint(0,1):
                child.nodes.append(copy(node2))
                continue
            child.nodes.append(copy(node1))

        return child

    def sameSpecie(self, n1,n2):
        # Lay out links:
        links1, links2 = {},{}
        for link in n1.links:
            links1[link.innov] = link
        for link in n2.links:
            links2[link.innov] = link

        # Lay out nodes:
        nodes1, nodes2 = {},{}
        for node in n1.nodes:
            nodes1[node.ID] = node
        for node in n2.nodes:
            nodes2[node.ID] = node

        # Evaluate topology overlap:
        disjointLinks = 0
        for i in range(0,self.link_innov+1):
            if (i in links1) ^ (i in links2):
                disjointLinks += 1

        # Evaluate link weight similarity:
        w_sum = 0
        w_coincident = 0
        for link1 in n1.links:
            if link1.innov in links2:
                w_sum += abs(link1.weight - links2[link1.innov].weight)
                w_coincident += 1

        # Evaluate node gene similarity:
        n_different = 0
        n_coincident = 0
        for node1 in n1.nodes:
            if node1.ID in nodes2:
                node2 = nodes2[node1.ID]
                n_coincident += 1
                if node1.func != node2.func:
                    n_different += 1
                w_sum += abs(node1.bias - node2.bias)
                w_coincident += 1

        a = disjointLinks / max(len(n1.links),len(n2.links))
        b = w_sum / w_coincident
        c = n_different / n_coincident

        if sqrt(a**2 + b**2 + c**2) > SimilarityThreshold:
            return False
        return True

pool = Pool()
