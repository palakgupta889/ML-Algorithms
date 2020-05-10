import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random


# dataset = {'k': [[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
# new_features = [5,7]

def k_nearest_neighbors(data,predict,k=3):
	if len(data) >= k:
		warnings.warn('K is set to a value less than total voting groups!')
	distances=[]
	for group in data:
		for features in data[group]:
			euclidean_dist = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidean_dist,group])

	votes = [i[1] for i in sorted(distances)[:k]]
	#print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1] / k

	return vote_result, confidence

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace=True)
df.drop(['id'],1,inplace=True)
full_data = df.astype(float).values.tolist()	#to treat all values as numbers bcoz apparently some values are loaded as strings from csv
random.shuffle(full_data)

test_size=0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
	train_set[i[-1]].append(i[:-1])

for i in test_data:
	test_set[i[-1]].append(i[:-1])

correct=0
total=0

for group in test_set:
	for data in train_set[group]:
		vote, confidence = k_nearest_neighbors(train_set, data, k=5) #bcoz scikit learn uses k=5 as default
		if group==vote:
			correct+=1
		else:
			print (confidence)
		total+=1

print('accuracy:', correct/total)
