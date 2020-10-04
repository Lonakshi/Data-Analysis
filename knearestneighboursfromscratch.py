import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings #warning the user for using dumb number for k
from matplotlib import style
from collections import Counter #
style.use('fivethirtyeight')

#euclidean_distance = sqrt( (plot1[0] - plot2[0]**2 + (plot[1]- plot2[1]**2) #original maths for calculating euclidean
                    
##print(euclidean_distance) 




#create a dataset mainly a dictionary maybe

dataset = { 'k': [[1,2], [2,3], [3,1]],'r': [[6,5],[7,7], [8,6]]} #features that correspond to the class of k
#we have two classes above and their features

new_features = [5,7] #we will know in future to which set is belong to


#for i in dataset:
    #for j in dataset[i]:
        #plt.scatter(j[0],j[1],s=100, color=i)
        #print(j[0],j[1])


#plt.scatter(new_features[0], new_features[1], s=100, color = 'g')
#plt.show()


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
        
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])

    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

result = k_nearest_neighbors(dataset,new_features,k=3)
print(result)


for i in dataset:
    for j in dataset[i]:
        plt.scatter(j[0],j[1],s=100, color=i)
        print(j[0],j[1])


plt.scatter(new_features[0], new_features[1], s=100, color = result)
plt.show()

























        
