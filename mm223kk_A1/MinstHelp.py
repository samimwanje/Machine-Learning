import numpy as np
import operator
from IPython.display import display

# Function used to run the k neareset neighbor for each training and test set.
def knn(dataSet, xCurrent, labels, trainingSet_Y, k, xS):
    neighbors = distanceCalc(dataSet,xCurrent,k)     # Tuple holding all the distances, with their index. Sorted by ascending order.
    return predict(neighbors, labels, trainingSet_Y, k, xS)               # Returns the majority if ok or fail.

# function used to run and calculate Euclidean distance.
def euclideanDistance(x1, xCurrent):
     distance = np.sum( (x1 - xCurrent)**2 , axis=0 ) #Euclidean distance formula euclideanDistance((x1, y), (x2, y2)) = √(x1 - x2)² + (y1 - y2)²
     return np.sqrt(distance) #Euclidean distance formula euclideanDistance((x1, y), (x2, y2)) = √(x1 - x2)² + (y1 - y2)²
   
 # Function used to Calculate the distance for every x in the training set
def distanceCalc(dataSet, xCurrent, k):
    distances = {}
    for x in range(len(dataSet)):
        dist = euclideanDistance(dataSet[x], xCurrent) # Euclidean distance from current x,y for current test set indexes.
        distances[x] = dist  # Add the distance into list.

    # Reorder tuples-array by distance ascending order. Index tuple maintain the same.
    ascendingDist = sorted(distances.items(), key=operator.itemgetter(1))

    # Add the k-nearset distances to an array.
    neighbors = []
    for x in range(k):
        neighbors.append(ascendingDist[x][0])
    
    return neighbors

# Function used to check if majority is ok or fail based on K.
def predict(neighbors, labels, trainingSet_Y, k, xS):
    ok = 0
    fail = 0
    for x in range(k):
        response = trainingSet_Y[neighbors[x]][0]
        if (response == labels[xS][0]):
            ok +=  1
        else:
            fail += 1
        
    if (ok > fail):
        return 1
    else:
        return 0


   