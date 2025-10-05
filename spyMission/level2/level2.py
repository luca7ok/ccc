import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dataset = pd.read_csv('data.csv')

x = dataset.iloc[:, 0:5].values

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(x_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=['pc1', 'pc2'])

standard_deviations = np.sqrt(pca.explained_variance_)

print(round(standard_deviations[0], 2), round(standard_deviations[1], 2))

pc1 = np.array(principal_components[:, 0])
pc2 = np.array(principal_components[:, 1])

plt.scatter(pc2, pc1, color='blue')
plt.savefig('scatter.png')
