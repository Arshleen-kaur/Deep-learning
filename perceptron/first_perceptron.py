# Perceptron is basically a binary classifier
# It works as a line or a plane or hyperplane to divide data into two regions
# Works only on linear or sort of linear dataset

# This is a basic perceptron implementation where no standardisation is done
# The perceptron accuracy is low 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron


df = pd.read_csv("placement.csv")

sns.scatterplot(data=df, x="cgpa", y="iq", hue="placement")
print(df.shape)
print(df.head())
print(df.tail())

# plt.savefig("scatterplot.png", dpi=300, bbox_inches="tight")  # Saves the image
# plt.show()  # Displays it

x= df.iloc[:,1:3]
y=df.iloc[:,-1]

p=Perceptron()

p.fit(x,y)
print(p.coef_)
print(p.intercept_)
print(p.score(x,y))
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(x.values, y.values, clf=p, legend=2)
plt.savefig("decision-boundary.png", dpi=300, bbox_inches="tight")
plt.show()

