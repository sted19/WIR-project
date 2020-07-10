from load_files import divide_dataset
from load_files import merge_dicts
from load_files import l_load
import correlation_coefficent 
import multiprocessing
import os
import numpy as np

from constants import *

'''
IMPORTANT READ ME: The books are represented as lists with the following format:
['uniqueISBN', 'bookTitle', 'bookAuthor', 'topic']
'''

# Dissimilarity between a pair of topics
def topic_dissimilarity(topic1, topic2):
    class1 = topic1.split(',')
    class2 = topic2.split(',')
    common_nodes = 0
    len1 = len(class1)
    len2 = len(class2)
    
    for i in range(min(len1, len2)):
        if class1[i] == class2[i]:
            common_nodes += 1
        else:
            break
    
    return 1 - 2*(common_nodes - 1)/(len1 -1 + len2 - 1)

# Consider boolean dissimilarity between a pair of authors
def author_dissimilarity(author1, author2):
    if author1 != author2:
        return 1
    else:
        return 0

# Input = actual top N list (where N > 10, such as 50 for ex.) and diversification factor (float in [0, 1]) that gives importance to the dissimilarity
# Output = top 10 list with elements in the top N passed in input, but more diversified (according to the value of the diversification factor)
def diversify_top_ten(old_list, diversification_factor):
    new_list = old_list[:1] # The first element remains always the same
    old_ranking = [list((rank, old_list[rank - 1], 0)) for rank in range(1, len(old_list) + 1)] # List of triples (rank(starting from 1), item, dissimilarity_actual_value)
    new_item = new_list[0]
    
    for i in range(1, 10):
        old_ranking = list(filter(lambda x: x[1][0] != new_item[0], old_ranking)) # Remove the element passed in the previous iteration to the new list from the old ranking
        dissimilarities = []
        
        for index in range(len(old_ranking)):
            triple = old_ranking[index]
            old_ranking[index][2] += (topic_dissimilarity(triple[1][3], new_item[3]) + author_dissimilarity(triple[1][2], new_item[2])) / 2 # Update the dissimilarity value considering the last added item (avg)
            dissimilarities.append(old_ranking[index])
        
        dissimilarities.sort(reverse=True, key=lambda tup: tup[2]) # Sort in decreasing order according to the dissimilarity value
        dissimilarity_rank = 0
        min_rank = len(old_list) + 1 + len(dissimilarities)
        
        for triple in dissimilarities:
            new_rank = triple[0] * (1 - diversification_factor) + dissimilarity_rank * diversification_factor # We consider the rank for both the old list and the dissimilarity sorted list
            if new_rank < min_rank:
                new_item = triple[1]
                min_rank = new_rank
            dissimilarity_rank += 1
        
        new_list.append(new_item)
    
    return new_list

def intra_list_similarity(l):
    similarity = 0
    n_items = len(l)

    for i in range(n_items - 1):
        for j in range(i + 1, n_items):
            (item1, item2) = (l[i], l[j])
            similarity += ((1 - topic_dissimilarity(item1[3], item2[3])) + (1 - author_dissimilarity(item1[2], item2[2]))) / 2

    return similarity

def lists_overlap(l1, l2):
    overlap = 0

    for i in range(len(l1)):
        if l1[i][0] == l2[i][0]:
            overlap += 1

    return overlap

    
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

    # Uncomment this part if you want to import the dictionary containing all infos about the several books
    '''
    with open('../Book-Crossing dataset cleaning/books_dict.pickle', 'rb') as handle:
        books_dict = pickle.load(handle)
    '''