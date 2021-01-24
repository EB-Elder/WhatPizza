import numpy as np
inputCols = 2
outputCols = 2
headArray = np.array([])
finalArray = np.array([])

filename = "TrainningData.csv"

print("Génération du dataset de test...")

for i in range(inputCols):
    headArray = np.append(headArray, "X"+str(i))
for i in range(outputCols):
    headArray = np.append(headArray, "Y"+str(i))
    
X = np.random.random((500, 2))
Y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else 0 for p in X])




for i in range(len(X)):
    finalArray = np.append(finalArray, np.concatenate([X[i], Y[i]]))
    
finalArray = np.append(headArray, finalArray)
finalArray = finalArray.reshape(int(len(finalArray)/(inputCols + outputCols)), inputCols + outputCols)

print("Fin de la génération !")

print("Écriture dans le fichier : " + filename)

np.savetxt(filename, finalArray, delimiter=";", fmt="%s")

print("Fin de l'écriture !")