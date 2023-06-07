import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the person's weight
xPos = 0
yPos = 0
c = 180

width = 1
height = 1
points = 100

# Define the vectors
A = np.array([-width, height, 0])    # top left
B = np.array([width, height, 0])     # top right
C = np.array([-width, -height, 0])   # bottom left
D = np.array([width, -height, 0])    # bottom right
P = np.array([xPos, yPos, c])

weightPerLeg = np.array([0.0, 0.0, 0.0, 0.0])

def calculateWeightOnLeg(xPos, yPos):
    
    P = np.array([xPos, yPos, c])

    dist_A = np.linalg.norm(P - A)
    dist_B = np.linalg.norm(P - B)
    dist_C = np.linalg.norm(P - C)
    dist_D = np.linalg.norm(P - D)
    
    distances = np.array([dist_A, dist_B, dist_C, dist_D])
    inverseDistances = np.array([0.0, 0.0, 0.0, 0.0])
    
    sum_inv_dists = 0
    for i in range (len(distances)):
        inverse = 1 / distances[i]
        inverseDistances[i] = inverse 
        sum_inv_dists += inverse     
    
    for i in range (len(distances)):
        weightPerLeg[i] = helperPerLeg(inverseDistances[i], sum_inv_dists)
    return weightPerLeg

def helperPerLeg(inv_dist, sum_inv_dists):
    weight = c * (inv_dist / sum_inv_dists)
    return weight

def weightArrayMaker(xRange, yRange):
    weight_A = np.zeros((len(xRange), len(yRange)))
    weight_B = np.zeros((len(xRange), len(yRange)))
    weight_C = np.zeros((len(xRange), len(yRange)))
    weight_D = np.zeros((len(xRange), len(yRange)))
    
    netWeight = np.zeros((len(xRange), len(yRange)))


    for i, x in enumerate(xRange):
        for j, y in enumerate(yRange):
            weightPerLeg = calculateWeightOnLeg(x, y)
            weight_A[i, j] = weightPerLeg[0]
            weight_B[i, j] = weightPerLeg[1]
            weight_C[i, j] = weightPerLeg[2]
            weight_D[i, j] = weightPerLeg[3]
            netWeight[i, j] = weight_A[i, j] + weight_B[i, j] + weight_C[i, j] + weight_D[i, j]
    return weight_A, weight_B, weight_C, weight_D

def plotterFunction():
    xRange = np.linspace(-width, width, points)
    yRange = np.linspace(-height, height, points)
    weight_A, weight_B, weight_C, weight_D = weightArrayMaker(xRange, yRange)
  
    X, Y = np.meshgrid(xRange, yRange)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, weight_A.T, cmap='coolwarm')
    ax.plot_surface(X, Y, weight_B.T, cmap='coolwarm')
    ax.plot_surface(X, Y, weight_C.T, cmap='coolwarm')
    ax.plot_surface(X, Y, weight_D.T, cmap='coolwarm')
    # ax.plot_surface(X, Y, netWeight.T, cmap='coolwarm')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Weight on Leg A')
    ax.set_title('Weight on Leg A')
    plt.show()

plotterFunction()