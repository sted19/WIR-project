import multiprocessing
import os

movies_path = os.path.join(os.path.join(os.getcwd(),'../Datasets'),'movies_dataset')
books_path = os.path.join(os.getcwd(),'../Book-Crossing dataset cleaning')

user_based_dict_path = os.path.join(movies_path,'utility_matrix_user_based.txt')
item_based_dict_path = os.path.join(movies_path,'utility_matrix_item_based.txt')

explicit_dict_path_books = os.path.join(books_path,'Explicit.csv')

implicit_dict_path_books = os.path.join(books_path,'Implicit.csv')

num_folds = 4

CORES = multiprocessing.cpu_count()

seed = 2

a = 0.95