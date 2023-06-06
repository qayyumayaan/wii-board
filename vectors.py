import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the person's weight
c = 180

# Define the vectors
A = np.array([-1, 1])
B = np.array([1, 1])
C = np.array([-1, -1])
D = np.array([1, -1])
P = np.array([0, 0, c])



def plotterFunction():
    x_range = np.linspace(-1, 1, 100)
    y_range = np.linspace(-1, 1, 100)
    weight_A = np.zeros((len(x_range), len(y_range)))

    
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            P = np.array([x, y])

    
    
    
    
    
    
    
    
    X, Y = np.meshgrid(x_range, y_range)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, weight_A.T, cmap='coolwarm')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Weight on Leg A')
    ax.set_title('Weight on Leg A')
    plt.show()

