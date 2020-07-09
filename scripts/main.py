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


        neigh_dict = utility_matrix.get(neighbor)

        if(neigh_dict == None):
            #print('user not present: {}'.format(neighbor))
            continue

        rating = neigh_dict.get(item)
        
        if  rating == None:
            continue

        numerator += rating*similarity
        denominator += similarity
    
    if denominator == 0:
        #print('denominator is 0, no user in the clique rated this item')
        return 0
    
    return numerator/denominator

"""
    returns the predicted rating given a user and an item

    user -> userID
    item -> itemID
    clique -> list of users that are similar to userID
"""

def predict_implicit(user, item, clique, utility_matrix):
    numerator = 0
    denominator = 0
    for elem in clique:
        neighbor = elem[0]
        similarity = elem[1]

        neigh_dict = utility_matrix.get(neighbor)
        denominator += similarity
        
        if(neigh_dict == None):
            continue
        rating = neigh_dict.get(item)
        
        if rating == None:
            continue
        numerator += similarity
    
    if denominator == 0:
        print('denominator is 0, no user in the clique')
        return 0
    return numerator/denominator

# if True it computes the tuning of the variable a
a_tuning = False 

if __name__ == "__main__":
    explicit_user_based_utility = l_load(explicit_dict_path_books, False)
    implicit_user_based_utility = l_load(implicit_dict_path_books, False)

    folds_explicit = divide_dataset(explicit_user_based_utility, 10)
    
    train_dict_explicit = merge_dicts([fold for fold in folds_explicit[:-1]])
    test_dict_expl = folds_explicit[-1] 

    train_dict_implicit = implicit_user_based_utility

    if(a_tuning):
        from constant_tuning import constant_tuning
        constant_tuning(train_dict_explicit, train_dict_implicit)
    
    

    """ user = list(test_dict.keys())[0]  

    print('user:{}'.format(user))

    clique = correlation_coefficent.compute_clique_with_implicit(user, train_dict_explicit, train_dict_implicit, 100, 1, 1) #TODO define a and b in a proper manner

    test_items = list(test_dict[user].keys())

    predictions = np.array([ [item, predict(user, item, clique, train_dict_explicit), test_dict[user][item]] for item in test_items])
    print(predictions) """