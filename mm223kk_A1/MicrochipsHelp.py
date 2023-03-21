import numpy as np
import operator 

 # Function used to Calculate the distance for every x in the training set
def distanceCalc(testSet,trainingSet,k):
    distances = {}
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testSet[0], testSet[1], trainingSet[x][0], trainingSet[x][1]) # Euclidean distance from current x,y for current test set indexes.
        distances[x] = dist  # Add the distance into list.

    # Reorder tuples-array by distance ascending order. Index tuple maintain the same.
    ascendingDist = sorted(distances.items(), key=operator.itemgetter(1))

    # Add the k-nearset distances to an array.
    neighbors = []
    for x in range(k):
        neighbors.append(ascendingDist[x][0])
    
    return neighbors

# Function used to check if majority is ok or fail based on K.
def okFail(neighbors, trainingSet):
    ok = 0
    fail = 0
    for x in range(len(neighbors)):
        response = trainingSet[neighbors[x]][2]
        if (response == 1):
            ok +=  1
        else:
            fail += 1
        
    if (ok > fail):
        return 1
    else:
        return 0


# function used to run and calculate Euclidean distance.
def euclideanDistance(x1, y1, x2, y2):
    distance = pow((x1 - x2), 2) + pow((y1 - y2),2)
    return np.sqrt(distance) #Euclidean distance formula euclideanDistance((x1, y), (x2, y2)) = √(x1 - x2)² + (y1 - y2)²


# Function used to run the k neareset neighbor for each training and test set.
def knn(trainingSet, testSet, k):
    neighbors = distanceCalc(testSet,trainingSet,k)     # Tuple holding all the distances, with their index. Sorted by ascending order.
    return okFail(neighbors, trainingSet)               # Returns the majority if ok or fail.


# This function is used to calculate the training errors.
def traningErrors(k, _trainingSet):

    errors = 0                            # Used to count how many errors in the result.
    chipY = _trainingSet[:,2]             # All rows column 2, fail or ok.
    chipY_np = np.asarray(chipY)          # Transfer to all rows column 2, fail or ok to np array. 

    index = 0
    for x in _trainingSet:

        neighbors = distanceCalc(x,_trainingSet,k)         # Tuple holding all the distances, with their index. Sorted by ascending order.
        class_result = okFail(neighbors, _trainingSet)     # Check the majority if ok or and then set class fail.

        # Check if error is found and increase error count.
        if (chipY_np[index] != class_result):
            errors += 1

        index += 1  # Go to next index.

    return errors


   