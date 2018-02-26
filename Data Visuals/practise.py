import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets

iris_dataset = datasets.load_iris()
X = iris_dataset.data
Y = iris_dataset.target
print(X)
print(Y)
iris_dataframe = pd.DataFrame(X, columns=iris_dataset.feature_names)

# Create a scatter matrix from the dataframe, color by y_train
grr = pd.plotting.scatter_matrix(iris_dataframe, c=Y, figsize=(5, 5), marker='o',
                                 hist_kwds={'bins': 20}, s=60, alpha=.8)

plt.show()
