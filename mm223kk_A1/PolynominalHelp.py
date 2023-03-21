import numpy as np
import operator 

# function used to run and calculate Euclidean distance.
def euclideanDistance(x1, x2):
    distance = (pow((x1 - x2), 2)) #Euclidean distance formula euclideanDistance((x1, y), (x2, y2)) = √(x1 - x2)² + (y1 - y2)²
    return np.sqrt(distance) 


# Function used to run the k neareset neighbor for each training and test set.
def knn(dataSet, xS, k):
    
    distances = {}
    for x in range(len(dataSet)):
        dist = euclideanDistance(xS, dataSet[x][0] ) # Euclidean distance for current x with x^2.
        distances[x] = dist  # Add the distance into list.

    # Reorder tuples-array by distance distance data in ascending order. Index (key-value)  maintains the same.
    ascendingDist = sorted(distances.items(), key=operator.itemgetter(1))
    
    # Add all the k-nearest indexes to neighbors array.
    neighbors = []
    for x in range(k):
        neighbors.append(ascendingDist[x][0])
        

    yValue = 0
    for x in range(k):
        yValue +=   dataSet[neighbors[x]][1]
    
    return (yValue/k) 