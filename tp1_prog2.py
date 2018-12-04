from sklearn import datasets, model_selection, neighbors
import numpy as np
import matplotlib.pyplot as plt

mnist = datasets.fetch_mldata('MNIST original')
index = np.random.randint(70000, size=5000)
data = mnist.data[index]
target = mnist.target[index]

# ==== Utiliser k = 10
# xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=0.8)

# k = 10
# clf = neighbors.KNeighborsClassifier(k)
# clf.fit(xtrain, ytrain)

# ==== Afficher la classe de l’image 4 et sa classe prédite
# image_4 = xtest[4].reshape(1,-1)
# print("predict : ", clf.predict(image_4))
# print(clf.predict_proba(image_4))
#
# plt.imshow(xtest.reshape((-1, 28, 28))[4],cmap=plt.cm.gray_r,interpolation="nearest")
# plt.show()

# ==== Affiche le score sur l’échantillon de test
# print(clf.score(xtest, ytest))

# ==== Faites varier le nombre de voisins (k) de 2 jusqu’à 15 et afficher le score. Quel est le k optimal
# print("k | scores\t\t\t\t\t\t\t\t| mean")
# for k in range(2, 15):
#
#     kf = model_selection.KFold(n_splits=10, shuffle=True)
#     scores = np.zeros(10)
#     i = 0
#
#     for train_index, test_index in kf.split(data):
#
#         # split dataset
#         X_train, X_test = data[train_index], data[test_index]
#         Y_train, Y_test = target[train_index], target[test_index]
#
#         # train with k-NN
#         clf = neighbors.KNeighborsClassifier(k)
#         clf.fit(X_train, Y_train)
#         scores[i] = clf.score(X_test, Y_test)
#         i += 1
#
#     print(k, ' | ', scores, ' | ', np.mean(scores))

# ==== Faites varier le pourcentage des échantillons (training et test) et afficher le score. Quel est le pourcentage remarquable
k = 10
clf = neighbors.KNeighborsClassifier(k)
print("train_size \t| score")
for train_size in range(0.5, 0.9, 0.1):
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size = train_size)

    kf = model_selection.KFold(n_splits=3, shuffle=True)
    scores = np.zeros(10)

    clf.fit(xtrain, ytrain)
    score = clf.score(xtest, ytest)

    print("")