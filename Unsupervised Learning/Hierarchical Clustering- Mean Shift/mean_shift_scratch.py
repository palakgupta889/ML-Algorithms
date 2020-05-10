import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

X,y = make_blobs(n_samples=15,centers=3,n_features=3)
# X = np.array([[1,2],[1.5,1.8],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],[8,2],[10,2],[9,3]])
# plt.scatter(X[:,0],X[:,1],s=150)
# plt.show()

colors=10*["g","r","c","b","k"]

class Mean_Shift:
	def __init__(self,radius=None,radius_norm_step = 100):
		self.radius_norm_step=radius_norm_step
		self.radius=radius

	def fit(self,data):

		if self.radius == None:
			all_data_centroid = np.average(data,axis=0)
			all_data_norm = np.linalg.norm(all_data_centroid)
			self.radius = all_data_norm/self.radius_norm_step

		weights = [i for i in range(self.radius_norm_step)][::-1]

		centroids={}
		for i in range(len(data)):
			centroids[i]=data[i]
		while True:
			new_centroids = []
			for i in centroids:
				in_bandwidth = []
				centroid = centroids[i]
				for feature_set in data:
					distance = np.linalg.norm(feature_set-centroid)
					if distance == 0:
						distance = 0.0000001
					weight_index = int(distance/self.radius)
					#if the point is outside the maximum distance boundary, i.e. more than 100 steps away, then set weight index to that max distance(100)
					if weight_index > self.radius_norm_step-1:	
						weight_index = self.radius_norm_step-1
					to_add = (weights[weight_index]**2)*[feature_set]
					in_bandwidth += to_add

				new_centroid = np.average(in_bandwidth,axis=0)
				new_centroids.append(tuple(new_centroid))
			uniques = sorted(list(set(new_centroids)))

			to_pop = []
			for i in uniques:
				for ii in uniques:
					if i==ii:
						break
					if np.linalg.norm(np.array(i)-np.array(ii))<=self.radius:
						to_pop.append(ii)

			for i in to_pop:
				try:
					uniques.remove(i)	
				except:
					pass		

			prev_centroids = centroids.copy()
			centroids = {}
			for i in range(len(uniques)):
				centroids[i]= np.array(uniques[i])

			optimized = True
			for i in centroids:
				if not np.array_equal(centroids[i],prev_centroids[i]):
					optimized= False
					break

			if optimized:
				break

		self.centroids=	centroids

		self.classifications = {}
		for i in range(len(self.centroids)):
			self.classifications[i] = []

		for feature_set in data:
			distances = [np.linalg.norm(feature_set-self.centroids[centroid]) for centroid in self.centroids]
			classification = distances.index(min(distances))
			self.classifications[classification].append(feature_set)

	def predict(self,data):
		distances = [np.linalg.norm(feature_set-self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification

clf = Mean_Shift()
clf.fit(X)
centroids = clf.centroids
#plt.scatter(X[:,0],X[:,1],s=150)
for classification in clf.classifications:
	color = colors[classification]
	for feature_set in clf.classifications[classification]:
		plt.scatter(feature_set[0],feature_set[1],marker='x',color=color, s=150, linewidth=5)

for c in centroids:
	plt.scatter(centroids[c][0],centroids[c][1], color = 'k', marker='*',s=150)
plt.show()


