import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from preprocess import PreProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

vectorizer = TfidfVectorizer()
pp = PreProcessor()
corpus = []

def elbow_cluster(n_max, data):
    WCSS = []
    for i in range(1, n_max):
        kmeans = KMeans(n_clusters=i, init="k-means++")
        kmeans.fit(data)
        WCSS.append(kmeans.inertia_)
        print("clustering with: " + str(i) + " centroids")
    return WCSS

def read_all_csv(departs):
    all_files = pd.DataFrame(columns=['link', 'number', 'text', 'cheie_depart'])
    for depart in departs:
        file = pd.read_csv('data/rapoarte' + str(depart) + '.csv', nrows=5000)
        file['cheie_depart'] = depart
        all_files = pd.concat([all_files, file], ignore_index=True)
    return all_files

def create_corpus(data):
    for row in range(0, data.shape[0]):
        raport = data['text'][row]
        raport = pp.to_lower(raport)
        raport = pp.remove_symbols(raport)
        #raport = pp.remove_diacritics(raport)
        raport = pp.remove_numbers(raport)
        raport = pp.remove_nonwords(raport)
        raport = pp.remove_diacritics(raport)
        raport = pp.remove_stopwords(raport)
        raport = pp.stem(raport)
        #raport = pp.lem(raport)
        corpus.append(raport)

departs = pd.read_csv('data/departamente.csv')
all_data = read_all_csv(departs.loc[:, 'cheie'])
all_data = all_data.sample(frac = 1).reset_index(drop=True)

create_corpus(all_data)
tfidf_matrix = vectorizer.fit_transform(corpus)

names = vectorizer.get_feature_names_out()
# with open('features.txt', 'w') as f:
#     for word in names:
#         f.write(word + '\n')

X_array = tfidf_matrix.toarray()
k_centroids = 6
clus = KMeans(n_clusters=k_centroids, init="k-means++")
clus.fit(X_array)

pca = PCA(n_components = 2)
reduced_f = pca.fit_transform(X_array)

reduced_cc = pca.transform(clus.cluster_centers_)
plt.scatter(reduced_f[:,0], reduced_f[:,1], c=clus.predict(X_array), marker='.', s=8)
plt.scatter(reduced_cc[:, 0], reduced_cc[:,1], marker='x', s=50, c='orange')
plt.show()

order_centroids = clus.cluster_centers_.argsort()[:, ::-1]
for i in range(k_centroids):
        print("Cluster %d:" % i, end="")
        for ind in order_centroids[i, :10]:
            print(" %s" % names[ind], end="")
        print()

# plt.plot(range(1, 14), elbow_cluster(14, X_array))
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()