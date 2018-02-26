import numpy as np
import pandas as pd
import pickle
from src import matrix_factorization_utilities

# Load user ratings
raw_dataset_df = pd.read_csv('../Data/ratings.csv')

raw_dataset_wo_timestamp_df = raw_dataset_df[['userId','movieId','rating']].copy()

print("raw_dataset_wo_timestamp: {}".format(raw_dataset_wo_timestamp_df))
# print("raw_dataset_df: {}".format(raw_dataset_df))

# Convert the running list of user ratings into a matrix
ratings_df = pd.pivot_table(raw_dataset_wo_timestamp_df, index='userId', columns='movieId', aggfunc=np.max)
print("ratings_df: {}".format(ratings_df.head(5)))


print("Starting U & M Calculations...")
# Apply matrix factorization to find the latest features
U, M = matrix_factorization_utilities.low_rank_matrix_factorization(ratings_df.as_matrix()
                                                                    , num_features=15
                                                                    , regularization_amount=0.1)
print("U and M Calculated")
# Find all predicted ratings by multiplying U and M
predicted_ratings = np.matmul(U, M)

print("U: {}".format(U[0]))
print("M: {}".format(M[0]))

# Save features and predicted ratings to files for later use
pickle.dump(U, open("../Data/user_features.dat", "wb"))
pickle.dump(M, open("../Data/product_features.dat", "wb"))
pickle.dump(predicted_ratings, open("../Data/predicted_ratings.dat", "wb"))

# Save all the ratings to a csv file
predicted_ratings_df = pd.DataFrame(index=ratings_df.index,
                                    columns=ratings_df.columns,
                                    data=predicted_ratings)
predicted_ratings_df.to_csv("../Data/predicted_ratings.csv")