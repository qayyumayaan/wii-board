import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def area(a, b, c):
    return abs((a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))/2.0)

# Define the vectors
A = np.array([-1, 1])
B = np.array([1, 1])
C = np.array([-1, -1])
D = np.array([1, -1])

# Define the person's weight
c = 180

# Define the range for x and y
x_range = np.linspace(-1, 1, 100)
y_range = np.linspace(-1, 1, 100)

# Initialize an array to store the weight on leg A for each (x, y) combination
weight_A = np.zeros((len(x_range), len(y_range)))

# def weightFinder(leg, x, y)

# Calculate the weight on leg A for each (x, y) combination
for i, x in enumerate(x_range):
    for j, y in enumerate(y_range):
        P = np.array([x, y])
        # Calculate the areas of triangles formed by the person and each set of three legs
        area_ABC = area(A, B, C)
        area_ABD = area(A, B, D)
        area_ACD = area(A, C, D)
        area_BCD = area(B, C, D)
        # Calculate the areas of triangles formed by the person and each set of two legs
        area_PBC = area(P, B, C)
        area_PBD = area(P, B, D)
        area_PCD = area(P, C, D)
        area_PAB = area(P, A, B)
        # Calculate the weight on each leg
        weight_A[i, j] = c * (area_BCD / (area_ABC + area_ABD + area_ACD + area_BCD))
        # weight_A[i,j] = x*y

# Create a 3D plot of weight on leg A
X, Y = np.meshgrid(x_range, y_range)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, weight_A.T, cmap='coolwarm')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Weight on Leg A')
ax.set_title('Weight on Leg A')
plt.show()
