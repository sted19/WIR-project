import multiprocessing
import os


books_path = os.path.join(os.getcwd(),'../../Datasets/BookDatasetCleaning')

explicit_dict_path_books = os.path.join(books_path,'Explicit.csv')

implicit_dict_path_books = os.path.join(books_path,'Implicit.csv')

num_folds = 4

CORES = multiprocessing.cpu_count()

seed = 2

a = 0.95 #importance of the explicit (0.05 for the implicit) used when building a clique by considering both

MIN_TEST_VALUE = 30
TO_DIFFERENTIATE_SIZE = 60
DIVERSIFICATION_FACTOR_RANGE = 10