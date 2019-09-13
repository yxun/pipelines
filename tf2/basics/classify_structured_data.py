#%% [markdown]
'''
- Load a CSV file using Pandas
- Build an input pipeline to batch and shuffle the rows using tf.data
- Map from columns in the CSV to features used to train the model using feature columns
- Build, train, and evaluate a model using Keras

    Dataset: a small dataset provided by the Cleveland Clinic Foudation for Heart Disease

'''


#%%

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


#%%
# Use Pandas to create a dataframe
URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
dataframe.head()


#%%
# Split the dataframe into train, validation, and test

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'valiation examples')
print(len(test), 'test examples')


#%%
# Create an input pipeline using tf.data

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


#%%
batch_size = 5      # A small batch size is used for demo
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


#%%
for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['age'])
    print('A batch of targets:', label_batch)


#%%
# Demo several types of feature columns
example_batch = next(iter(train_ds))[0]


#%%
# A utility method to create a feature column
# and to transform a batch of data
def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())


#%%
age = feature_column.numeric_column("age")
demo(age)

#%%
# Bucketized columns

age_buckets = feature_column.bucketized_column(age, boundaries=[
        18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)

#%%
# Categorical columns

thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible']
)

thal_one_hot = feature_column.indicator_column(thal)
demo(thal_one_hot)


#%%
# Embedding columns

# Using an embedding column is best when a categorical column has many possible values.

thal_embedding = feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)


#%%
# Hashed feature columns

thal_hashed = feature_column.categorical_column_with_hash_bucket(
    'thal', hash_bucket_size=1000
)
demo(feature_column.indicator_column(thal_hashed))


#%%
# Crossed feature columns

crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(feature_column.indicator_column(crossed_feature))

#%%
# Choose which columns to use

feature_columns = []

# numeric cols
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))

# bucketized cols
age_buckets = feature_column.bucketized_column(age, boundaries=[
    18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator cols
thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible']
)
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding cols
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)


#%%
# Create a feature layer

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

#%%
# Use a larger batch size

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


#%%
# Create, compile and train the model

model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'],
            run_eagerly=True)

model.fit(train_ds,
        validation_data=val_ds,
        epochs=5)


#%%
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

# Note: To improve accuracy, think carefully about which features to include in your model, 
# and how they should be represented.
# Use a larger and more complex datasets.


#%%
