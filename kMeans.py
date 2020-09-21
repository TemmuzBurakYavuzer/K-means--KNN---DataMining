import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1, 2],[0.5, 1.8],[10, 8 ],[8, 8],[5, 6],[9,11],[1,2],[6,9],[0,3],[5.7,4],[6,4.9],[4,7],[6,3]])

colors = ["g","r","c","b","k"]

class K_Means:
    def __init__(self, k=3, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

#her iterasyonda clasificationsları temizliyoruz k sayısı aynı kalıyor ama cendroidin degerleri değişiyor
    def fit(self,data):
        self.centroids = {}
        # ilk i sayıdaki degerler bizim ilk centroidimiz
        for i in range(self.k):
            self.centroids[i] = data[i]
        #optimizasyon
        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []
            #boş set i populate ediyoruz
            for featureset in data:
                #olusturulan list k sayıda degerle populate ediliyor
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)
            #butun classificationların ortalamasını alıcak boylece centroidleri yeniden tanımlıycaz
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True
            #centroidlerin hareketleri torelans dan fazla ise optimize değil
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="p", color=color, s=50, linewidths=5)

plt.show()