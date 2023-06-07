import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the person's weight
xPos = 0
yPos = 0
c = 180

# Define the vectors
A = np.array([-1, 1, 0])
B = np.array([1, 1, 0])
C = np.array([-1, -1, 0])
D = np.array([1, -1, 0])
P = np.array([xPos, yPos, c])

weightPerLeg = np.array([0.0, 0.0, 0.0, 0.0])

def calculateWeightOnLeg(xPos, yPos):
    
    P = np.array([xPos, yPos, c])

    # Calculate the distances from the person to each leg
    dist_A = np.linalg.norm(P - A)
    dist_B = np.linalg.norm(P - B)
    dist_C = np.linalg.norm(P - C)
    dist_D = np.linalg.norm(P - D)

    # Calculate the inverse of these distances
    inv_dist_A = 1 / dist_A
    inv_dist_B = 1 / dist_B
    inv_dist_C = 1 / dist_C
    inv_dist_D = 1 / dist_D
    
    sum_inv_dists = inv_dist_A + inv_dist_B + inv_dist_C + inv_dist_D

    weightPerLeg = helperPerLeg(inv_dist_A, sum_inv_dists)
    return weightPerLeg

def helperPerLeg(inv_dist, sum_inv_dists):
    weight = c * (inv_dist / sum_inv_dists)
    return weight

def plotterFunction():
    xRange = np.linspace(-1, 1, 100)
    yRange = np.linspace(-1, 1, 100)
    weight_A = np.zeros((len(xRange), len(yRange)))

    # P = np.array([xRange, yRange])

    for i, x in enumerate(xRange):
        for j, y in enumerate(yRange):
            weightPerLeg = calculateWeightOnLeg(x, y)
            weight_A[i, j] = weightPerLeg

    
    X, Y = np.meshgrid(xRange, yRange)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, weight_A.T, cmap='coolwarm')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Weight on Leg A')
    ax.set_title('Weight on Leg A')
    plt.show()

plotterFunction()