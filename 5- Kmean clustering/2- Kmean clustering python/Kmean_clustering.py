import numpy as np
import matplotlib.pyplot as plt

class KMean:

    def __init__(self, k=5, n_iteration=40000,plot_in_optimization=True):
        self.k = k
        self.n_iteration = n_iteration
        self.plot_in_optimization=plot_in_optimization
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []

    def fit(self,X):
        # Kmean clustering is considered as unsupervised model i.e., no "y"
        # No fitting method is applied in Kmean, so we skipp this stage
        pass

    def predict(self,X):

        # 1- Centroid initialization
        self.X=X
        self.n_sampels,self.n_faetures=X.shape
        random_centroids_indices=np.random.choice(self.n_sampels,self.k,replace=False)
        self.centroids=[self.X[idx] for idx in random_centroids_indices]

        # 2- Optimization
        for _ in range(self.n_iteration):
            point=[]
            # 2.1- Identifying clusters
            self.clusters=self._make_cluster(self.centroids)
            if self.plot_in_optimization:
                self.plot()


            # 2.2- Identifying new centroids
            old_centroid=self.centroids

            self.centroids=self._make_new_centroid(self.clusters)
            if self.plot_in_optimization:
                self.plot()

            # 2.3- Check if converged
            if self._is_converged(old_centroid,self.centroids):
                break


        # 3- Return cluster's labels
        return self._create_clusters_labels(self.clusters)

    def _create_clusters_labels(self,clusters):
        labels=np.empty(self.n_sampels)
        for cidx,cluster in enumerate(clusters):
            for x in cluster:
                labels[x]=cidx

        return labels



    def _make_cluster(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, x in enumerate(self.X):
            centroid_idx = [self._euclidean_distance(x, centroid) for centroid in centroids]
            nearest_centroid = np.argmin(centroid_idx)
            clusters[nearest_centroid].append(idx)


        return clusters

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))


    def _make_new_centroid(self,clusters):
        centroids = np.zeros((self.k, self.n_faetures))
        for cidx, cluster in enumerate(self.clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cidx] = cluster_mean


        return centroids

    def _is_converged(self,old_centroid,centroids):
        centroids_difference=[self._euclidean_distance(old_centroid[i],centroids[i]) for i in range(self.k)]
        return (sum(centroids_difference)==0)

    def plot(self):
        fig,ax=plt.subplots(figsize=(7,4))
        for idx,cluster in enumerate(self.clusters):
            data_points=self.X[cluster].T # transpose is important to have  columns for plotting
            ax.scatter(*data_points)

        for centroid in self.centroids:
            ax.scatter(*centroid,marker='o',color='black',linewidth=5)
        plt.show()
