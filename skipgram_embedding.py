import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Assuming you have your entire df_person DataFrame

# Define embedding dimensions
embedding_dim = 2

# Generate a random key
key = jax.random.PRNGKey(0)

# Create a dictionary to store the embeddings
embeddings = {}

# Select a subset of columns for embedding and visualization
columns_subset = ['month_of_birth']


# Get number of categories before one-hot encoding 
num_categories = len(df_person[column].unique())

# Iterate over each selected column
for column in columns_subset:
    if df_person[column].dtype == 'object':
        # Perform one-hot encoding for categorical columns
        encoded_column = pd.get_dummies(df_person[column], dtype=np.int8)
        encoded_column = encoded_column.values
        encoded_column = encoded_column.astype(np.int8)
    else: 
        # For numerical columns, use the values directly
        encoded_column = df_person[column].values 

    encoded_column = encoded_column.astype(jnp.int32)

    # Randomly initialize the embedding matrix  
    embedding_matrix = jax.random.normal(key, (num_categories, embedding_dim)) 

    # Perform embedding lookup for each value in the column
    embedded_column = jnp.take(embedding_matrix, encoded_column, axis=0) 

    # Store the embeddings for the column in the dictionary
    embeddings[column] = embedded_column
