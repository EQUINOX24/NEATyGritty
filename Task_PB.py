import pygame
import math
import random
from NEATyGritty import Act

InOut = { 'in': 1, 'out': 1, 'activations' : [Act.TANH]}


tau = 2*math.pi
half_pi = math.pi/2

white = (255, 255, 255)
gray = (174, 174, 174)
green = (94, 217, 94)
blue = (94, 113, 217)
red = (217, 44, 44)
purple = (113, 44, 174)
black = (0, 0, 0)

color = [gray, green, blue, red, purple]

varKeep = { 'theta_0' : 0.0, 'c' : 0 }


pygame.init()
disp_h, disp_w = 512, 512
gameDisplay = pygame.display.set_mode((disp_w, disp_h))
pygame.display.set_caption('Press s to show Pole Balancing')
clock = pygame.time.Clock()

quart = disp_h/4
def X(x):
    return round(quart*(2 + x))
def Y(y):
    return round(quart*(2 - y))
def L(l):
    return quart*l


class Phys():
    # physical parameters
    g = 9.8 # free fall acceleration
    l = 0.3 # pole length
    m_c = 35.0 # cart mass
    # numeric parameters
    h = 1/30 # delta t
    # optimization parameters
    b = 1.0
    c = 1.0
    # state variables
    n0 = True
    theta_n0 = 0.0
    theta_n1 = 0.0
    omega_n0 = 0.0
    omega_n1 = 0.0
    # output state variables
    theta = 0.0
    def __init__(self):
        self.n0 = True
        self.theta_n0 = varKeep['theta_0']
        self.theta = self.theta_n0
        self.omega_n0 = 0.0

        self.b = self.h*self.g/self.l
        self.c = 1/(self.g*self.m_c)

    def SetParams(self, g=False, l=False, m_c=False, h=False):
        if g:
            self.g = g
        if l:
            self.l = l
        if m_c:
            self.m_c = m_c
        if h:
            self.h = h
        self.b = self.h*self.g/self.l
        self.c = 1/(self.g*self.m_c)

    def restIV(self): # rest initial values
        self.n0 = True
        self.theta_n0 = varKeep['theta_0']
        self.theta = self.theta_n0
        self.omega_n0 = 0.0

    def PhysStep(self,F=0.0):
        if self.n0:
            self.n0 = False
            self.theta_n1 = self.theta_n0 + self.h*self.omega_n0
            self.omega_n1 = self.omega_n0 + self.b*(math.sin(self.theta_n1) - self.c*F*math.cos(self.theta_n1))
            self.theta = self.theta_n1
        else:
            self.n0 = True
            self.theta_n0 = self.theta_n1 + self.h*self.omega_n1
            self.omega_n0 = self.omega_n1 + self.b*(math.sin(self.theta_n0) - self.c*F*math.cos(self.theta_n0))
            self.theta = self.theta_n0

s = 0.03
def Randomize():
    theta = random.gauss(0, s)
    if abs(theta) <= 2*s/5:
        theta += (4*s/5)*(random.randint(0,1)-1/2)
    if abs(theta) > 8*s:
        theta = 8*s*(random.randint(0,1)-1/2)
    varKeep['theta_0'] = theta

Randomize()
PB = Phys()
f = 90.0

Show = { 'value' : False }
N = 15000
def Evaluator(NN):
    # variables
    Force = 0.0

    varKeep['c'] += 1
    varKeep['c'] %= len(color)
    for n in range(0,N):
        # events
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    if Show['value']:
                        Show['value'] = False
                    else:
                        Show['value'] = True
        # computation
        Out = NN.Evaluate([PB.theta])[0]
        if Out > 0.333:
            Force = f
        elif Out < -0.333:
            Force = -f
        else:
            Force - 0.0
        PB.PhysStep(F=Force)

        if abs(PB.theta) > half_pi:
            break

        x = math.sin(-PB.theta)
        y = math.cos(PB.theta)

        if Show['value']:
            # display
            gameDisplay.fill(white)

            pygame.draw.line(gameDisplay, black, (X(0), Y(0)), (X(x), Y(y)), 5)
            pygame.draw.rect(gameDisplay, color[varKeep['c']], (X(-0.5), Y(0), L(1), L(0.25)))

            pygame.display.update()

            clock.tick(60)
    PB.restIV()
    NN.fitness = n/(N-1)
    return NN.fitness
