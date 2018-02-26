import numpy as np
import pandas as pd

raw_ratings_df = pd.read_csv('../Data/ratings.csv')
raw_ratings_df = raw_ratings_df[['userId','movieId','rating']].copy()
msk = np.random.rand(len(raw_ratings_df)) > 0.80
train = raw_ratings_df[msk]
test = raw_ratings_df[~msk]

train.to_csv('../Data/ratings_set_training.csv')
test.to_csv('../Data/ratings_set_testing.csv')

print("Training Data: {}".format(train))
print("Testing Data:  {}".format(test))
