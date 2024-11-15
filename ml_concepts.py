"""
Use sklearn to build a simple linear regression model.
Explain how to prevent overfitting in a decision tree.
Write a function to calculate cosine similarity between two vectors.
Use Faiss to perform nearest neighbor search on a dataset of embeddings.
Split a dataset into training and test sets, and train a logistic regression model.
Evaluate the performance of a classification model using a confusion matrix.
Implement k-fold cross-validation for a random forest classifier.
Use a pretrained word embedding model (e.g., Word2Vec) for a text similarity task.
Discuss how to handle imbalanced datasets in classification tasks.
Optimize hyperparameters of an SVM using grid search.
"""

# Works with faiss env


# Use sklearn to build a simple linear regression model.

import polars as pl
import pandas as pd

df = pl.from_pandas(pd.read_csv('data/employment_data.csv'))
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])

# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)


# Explain how to prevent overfitting in a decision tree.

"""
Decision trees are approximating using depth of leaves. The more depth in a tree the more complex
the decision and function. To prevent overfitting one should set the max_depth to a low number. 
To fully assess overfitting, one divide the dataset into train and test. Check the accuracy/MSE
for different max depths, and see where the model is not improving on the test set. This would ensure
no overfitting.

"""


# Write a function to calculate cosine similarity between two vectors.

vector_1 = np.array([1,3,4,5,6])
vector_2 = np.array([1,2,1,1,1])
assert len(vector_1) == len(vector_2)

cosine_sim = np.dot(vector_1,vector_2)/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))
print(cosine_sim)


# Use Faiss to perform nearest neighbor search on a dataset of embeddings.
import requests
from io import StringIO
res = requests.get('https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/sick2014/SICK_train.txt')
# create dataframe
data = pd.read_csv(StringIO(res.text), sep='\t')

sentences = data['sentence_A'].tolist()

print()

from sentence_transformers import SentenceTransformer
# initialize sentence transformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')
# create sentence embeddings
sentence_embeddings = model.encode(sentences)
sentence_embeddings.shape

import faiss
d = sentence_embeddings.shape[1]

index = faiss.IndexFlatL2(d)
index.add(sentence_embeddings)

k = 4
xq = model.encode(["Someone sprints with a football"])
D, I = index.search(xq, k)  # search
print(I)
data['sentence_A'].iloc[I[0]]


# Split a dataset into training and test sets, and train a logistic regression model.

from sklearn.model_selection import train_test_split

y = df['Data_value'].fill_nan(df['Data_value'].mean()).fill_null(df['Data_value'].mean())
X = df[['Period', 'Magnitude']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

y.filter(y.is_null())
X['Period'].filter(X['Period'].is_nan())

reg = LinearRegression().fit(X_train, y_train)
preds = reg.predict(X_test)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, preds))

# Evaluate the performance of a classification model using a confusion matrix.





# Implement k-fold cross-validation for a random forest classifier.


