from src import similarities_mod as sm

numReturn = 5
sorted_movie_list = sm.getTopN(2, numReturn)

print(sorted_movie_list[['title', 'difference_score']][0:numReturn])
