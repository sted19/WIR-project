from load_files import divide_dataset
from load_files import merge_dicts
from load_files import l_load
import correlation_coefficent 
import multiprocessing
import os
import numpy as np

from constants import *

"""
    returns a value that represents
    how similar two classes are
"""
def book_distance(book1, book2, book_data):
    class1 = book_data[book1]["product_category"].split(',')
    class2 = book_data[book2]["product_category"].split(',')

    last_common = 0

    for i in range(0, min(len(class1), len(class2))):
        if(class1[i]!=class2[i]):
            last_common = i - 1 
    
    return len(class1) + len(class2) - 2*last_common 

    
"""
    returns the predicted rating given a user and an item

    user -> userID
    item -> itemID
    clique -> list of users that are similar to userID
"""

def predict(user, item, clique, utility_matrix):
    numerator = 0
    denominator = 0
    for elem in clique:
        neighbor = elem[0]
        similarity = elem[1]

        neigh_dict = utility_matrix[neighbor]
        rating = neigh_dict.get(item)
        
        if  rating == None:
            continue

        numerator += rating*similarity
        denominator += similarity
    
    if denominator == 0:
        print('denominator is 0, no user in the clique rated this item')
        return -1
    
    return numerator/denominator



"""
    returns the K items with higher score for 
    the user 
""" 

if __name__ == "__main__":
    explicit_user_based_utility = l_load(explicit_dict_path_books, False)
    implicit_user_based_utility = l_load(implicit_dict_path_books, False)

    folds_explicit = divide_dataset(explicit_user_based_utility, num_folds)
    folds_implicit = divide_dataset(implicit_user_based_utility, num_folds)

    train_dict_explicit = merge_dicts([fold for fold in folds_explicit[:3]])
    test_dict = folds_explicit[3] 

    train_dict_implicit = merge_dicts([fold for fold in folds_explicit[:3]])
    test_dict_impl = folds_implicit[3] 

    user = list(test_dict.keys())[0]  

    print('user:{}'.format(user))

    clique = correlation_coefficent.compute_clique_with_implicit(user, train_dict_explicit, train_dict_implicit, 100, 1, 1) #TODO define a and b in a proper manner

    test_items = list(test_dict[user].keys())

    predictions = np.array([ [item, predict(user, item, clique, train_dict_explicit), test_dict[user][item]] for item in test_items])
    print(predictions)