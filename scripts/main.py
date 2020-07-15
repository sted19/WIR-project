from load_files import divide_dataset
from load_files import merge_dicts
from load_files import l_load
from load_files import build_books_data
import correlation_coefficent 
import multiprocessing
import os
import numpy as np
import pickle

from constants import *

np.random.seed(seed)

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
            old_ranking[index][2] += 0.75 * topic_dissimilarity(triple[1][3], new_item[3]) + 0.25 * author_dissimilarity(triple[1][2], new_item[2]) # Update the dissimilarity value considering the last added item (weighted avg)
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
            similarity += 0.75 * (1 - topic_dissimilarity(item1[3], item2[3])) + 0.25 * (1 - author_dissimilarity(item1[2], item2[2]))

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

'''
    computes precision and recall
    predicted_ranking -> [[item_id, ... ]]
    true_ratings -> {item_id:rating}
'''
def compute_implicit_value(predicted_ranking, true_ratings, scale=1):
    total_relevant = 0
    returned_relevant = 0
    threshold = 5 * scale
        
    for item in true_ratings.keys():
        if(true_ratings[item] > threshold):
            total_relevant += 1

    for item_ in predicted_ranking:
        item_id = item_[0]

        if(true_ratings[item_id] > threshold): 
            returned_relevant += 1

    if(total_relevant == 0):
        recall = 1
    else:
        recall = returned_relevant/total_relevant
        
    precision = returned_relevant/len(predicted_ranking)

    return (precision, recall)

'''
    Paramenters
        predicted_ranking -> [[item_id, ... ]] sorted from the first to return, to the last one
        true_ratings -> {item_id:rating}
    computes the list value and the difference between the optimal and the computed one
'''
def compute_list_values(predicted_ranking, true_ratings):
    (opt_list_value, list_value) = (0, 0)
    opt_list = list(true_ratings.values())
    opt_list.sort(reverse = True)
    opt_list = opt_list[:10]

    for item_idx in range (len(predicted_ranking)):
        true_rating = true_ratings[predicted_ranking[item_idx][0]]
        score = (1/(np.log(item_idx+np.e)))*true_rating

        opt_true_rating = opt_list[item_idx]
        opt_score = (1/(np.log(item_idx+np.e))) * opt_true_rating

        list_value += score
        opt_list_value += opt_score

    return (list_value, opt_list_value - list_value)

'''
    parameters
        train_dict_explicit -> {user:{item:explicit_rating}}
        train_dict_implicit -> {user:{item:implicit_rating}}
        test_dict_expl -> {user:{item:explicit_rating}}
        all_imp_items -> set of all the items

    returns {diversification_factor:{precision, recall, list_value}}
'''
def user_binary_tests(train_dict_explicit, train_dict_implicit, test_dict_expl, test_dict_impl, all_imp_items):
    results = {}
    book_data = build_books_data(explicit_dict_path_books)

    test_users = set(test_dict_expl.keys()).intersection(set(test_dict_impl.keys())).intersection(set(train_dict_explicit.keys())).intersection(set(train_dict_implicit.keys()))


    num_tests = 0

    for user in test_users:
        tmp_user_prediction_list = []

        if(len(set(test_dict_impl[user].keys())) >= MIN_TEST_VALUE):

            additiona_test_items = list(all_imp_items.difference(set(test_dict_impl[user].keys())))[:int(TO_DIFFERENTIATE_SIZE/2)]

            tmp_test_items = set(test_dict_impl[user].keys()).union(additiona_test_items)

            tmp_impl_test_dict = {}

            for item in tmp_test_items:
                if(book_data.get(item) == None or train_dict_explicit[user].get(item) != None or train_dict_implicit[user].get(item)!=None):
                    continue
                elif(test_dict_impl[user].get(item) != None or (test_dict_expl[user].get(item) != None and test_dict_expl[user].get(item) > 6)):
                    tmp_impl_test_dict[item] = 1
                else:
                    tmp_impl_test_dict[item] = 0


            num_tests += 1
            print('new test user: {}'.format(num_tests))

            clique = correlation_coefficent.compute_clique_with_implicit(user, train_dict_explicit, train_dict_implicit, 100, a, 1-a, save=False)

            for item in tmp_impl_test_dict.keys():
                tmp_rating = predict(user, item, clique, train_dict_explicit)
                tmp_user_prediction_list.append([item, tmp_rating])

            prediction_list = np.array(tmp_user_prediction_list)
            prediction_list = prediction_list[np.argsort(prediction_list[:,1])[::-1]][0:TO_DIFFERENTIATE_SIZE]

            item_topic_book_list = []
            for item_ in  prediction_list:
                item_id = item_[0]
                item_topic_book_list.append([item_id, book_data[item_id]['bookTitle'], 
                                                book_data[item_id]['bookAuthor'], 
                                                book_data[item_id]['topic']])
            
            for i in range(DIVERSIFICATION_FACTOR_RANGE):
                rate = i/DIVERSIFICATION_FACTOR_RANGE
                
                diversify_top_ten_list = diversify_top_ten(item_topic_book_list, rate) 

                #list_value = compute_list_values(diversify_top_ten_list, test_dict_expl[user])
                
                precision, recall = compute_implicit_value(diversify_top_ten_list, tmp_impl_test_dict, 0.1)

                if(results.get(i) == None):
                    results[i] = {}
                    #results[i]['list_value'] = list_value
                    results[i]['precision'] = precision
                    results[i]['recall'] = recall
                else :
                    #results[i]['list_value'] = results[i]['list_value'] + list_value
                    results[i]['precision'] = results[i]['precision'] + precision
                    results[i]['recall'] = results[i]['recall'] + recall

    for test in results.keys():
        #results[test]['list_value'] = results[test]['list_value']/num_tests
        results[test]['precision'] = results[test]['precision']/num_tests
        results[test]['recall'] = results[test]['recall']/num_tests

    return results

def user_expl_tests(train_dict_explicit, train_dict_implicit, test_dict_expl):
    results = {}
    book_data = build_books_data(explicit_dict_path_books)

    num_tests = 0

    for user in test_dict_expl.keys():
        tmp_user_prediction_list = []

        if(len(set(test_dict_expl[user].keys())) >= MIN_TEST_VALUE):
            num_tests += 1
            print('new test user: {}'.format(num_tests))

            clique = correlation_coefficent.compute_clique_with_implicit(user, train_dict_explicit, train_dict_implicit, 100, a, 1-a, save=False)

            for item in test_dict_expl[user].keys():
                tmp_rating = predict(user, item, clique, train_dict_explicit)
                tmp_user_prediction_list.append([item, tmp_rating])

            prediction_list = np.array(tmp_user_prediction_list)
            prediction_list = prediction_list[np.argsort(prediction_list[:,1])[::-1]][0:int(TO_DIFFERENTIATE_SIZE/2)]

            item_topic_book_list = []
            for item_ in  prediction_list:
                item_id = item_[0]
                item_topic_book_list.append([item_id, book_data[item_id]['bookTitle'], 
                                                book_data[item_id]['bookAuthor'], 
                                                book_data[item_id]['topic']])

            for i in range(DIVERSIFICATION_FACTOR_RANGE):
                rate = i/DIVERSIFICATION_FACTOR_RANGE

                diversify_top_ten_list = diversify_top_ten(item_topic_book_list, rate) 

                (list_value, difference) = compute_list_values(diversify_top_ten_list, test_dict_expl[user])

                precision, recall = compute_implicit_value(diversify_top_ten_list, test_dict_expl[user])

                if(results.get(i) == None):
                    results[i] = {}
                    results[i]['list_value'] = list_value
                    results[i]['precision'] = precision
                    results[i]['recall'] = recall
                    results[i]['list_value_difference'] = difference
                else :
                    results[i]['list_value'] = results[i]['list_value'] + list_value
                    results[i]['precision'] = results[i]['precision'] + precision
                    results[i]['recall'] = results[i]['recall'] + recall
                    results[i]['list_value_difference'] = results[i]['list_value_difference'] + difference

    for test in results.keys():
        results[test]['list_value'] = results[test]['list_value']/num_tests
        results[test]['precision'] = results[test]['precision']/num_tests
        results[test]['recall'] = results[test]['recall']/num_tests
        results[test]['list_value_difference'] = results[test]['list_value_difference']/num_tests

    return results


def invert_dict(item_based_dict):
    user_based_dict = {}
    
    for item in item_based_dict.keys():
        item_dict = item_based_dict[item]
        
        for user in item_dict.keys():
            if(user_based_dict.get(user) == None):
                user_based_dict[user] = {}
            user_based_dict[user][item] = item_based_dict[item][user]

    return user_based_dict

# Here we pass another parameter because we need also the dictionary {userId: {itemId: rating}} that in the user_tests is present by default
def item_tests(train_dict_explicit, train_dict_implicit, test_dict_expl, user_based_test_dict_expl):
    results = {}
    book_data = build_books_data(explicit_dict_path_books)
    users_prediction_lists = {}

    num_tests = 0

    for item in test_dict_expl.keys():
        if(train_dict_explicit.get(item) == None):
            continue
        clique = None

        for user in test_dict_expl[item].keys():
            if(len(user_based_test_dict_expl[user].keys()) >= MIN_TEST_VALUE):
                
                if(users_prediction_lists.get(user) == None):
                    users_prediction_lists[user] = []
                    num_tests += 1
                    print('new test user: {}'.format(num_tests))
                
                if(clique is None):
                    clique = correlation_coefficent.compute_clique_with_implicit(item, train_dict_explicit, train_dict_implicit, 100, a, 1-a, save=False)

                tmp_rating = predict(item, user, clique, train_dict_explicit)
                users_prediction_lists[user].append([item, tmp_rating])
    
    print("Loop on items ended")
    
    for user in users_prediction_lists.keys():
        prediction_list = np.array(users_prediction_lists[user])
        prediction_list = prediction_list[np.argsort(prediction_list[:,1])[::-1]][0:int(TO_DIFFERENTIATE_SIZE/2)]

        item_topic_book_list = []
        for item_ in  prediction_list:
            item_id = item_[0]
            item_topic_book_list.append([item_id, book_data[item_id]['bookTitle'], book_data[item_id]['bookAuthor'], book_data[item_id]['topic']])
            
        for i in range(DIVERSIFICATION_FACTOR_RANGE):
            rate = i/DIVERSIFICATION_FACTOR_RANGE
            
            diversify_top_ten_list = diversify_top_ten(item_topic_book_list, rate) 

            (list_value, difference) = compute_list_values(diversify_top_ten_list, user_based_test_dict_expl[user])
            
            precision, recall = compute_implicit_value(diversify_top_ten_list, user_based_test_dict_expl[user])

            if(results.get(i) == None):
                results[i] = {}
                results[i]['list_value'] = list_value
                results[i]['precision'] = precision
                results[i]['recall'] = recall
                results[i]['list_value_difference'] = difference

            else :
                results[i]['list_value'] = results[i]['list_value'] + list_value
                results[i]['precision'] = results[i]['precision'] + precision
                results[i]['recall'] = results[i]['recall'] + recall
                results[i]['list_value_difference'] = results[i]['list_value_difference'] + difference

    for test in results.keys():
        results[test]['list_value'] = results[test]['list_value']/num_tests
        results[test]['precision'] = results[test]['precision']/num_tests
        results[test]['recall'] = results[test]['recall']/num_tests
        results[test]['list_value_difference'] = results[test]['list_value_difference']/num_tests

    return results

# if True computes the tuning of the variable a
a_tuning = False 
# if True computes the user-based binary test 
user_binary_tests_cond = False
# if True computes the user-based explicit test 
user_expl_tests_cond = False
# if True computes the item-based test
item_tests_cond = True

if __name__ == "__main__":
    explicit_user_based_utility = l_load(explicit_dict_path_books, item_tests_cond)
    implicit_user_based_utility = l_load(implicit_dict_path_books, item_tests_cond)

    folds_explicit = divide_dataset(explicit_user_based_utility, 3)
    
    train_dict_explicit = merge_dicts([fold for fold in folds_explicit[:-1]])
    test_dict_expl = folds_explicit[-1]
    
    if(user_binary_tests_cond):
        all_imp_items = set((invert_dict(implicit_user_based_utility)).keys())
        folds_implicit = divide_dataset(implicit_user_based_utility, 3)
        train_dict_implicit = merge_dicts([fold for fold in folds_implicit[:-1]])
        test_dict_impl = folds_implicit[-1]
        res = user_binary_tests(train_dict_explicit, train_dict_implicit, test_dict_expl, test_dict_impl, all_imp_items)
        print(res)

    train_dict_implicit = implicit_user_based_utility

    if(user_expl_tests_cond):
        res = user_expl_tests(train_dict_explicit, train_dict_implicit, test_dict_expl)
        print(res)

    if(item_tests_cond):
        user_based_test_dict_expl = invert_dict(test_dict_expl) 
        res = item_tests(train_dict_explicit, train_dict_implicit, test_dict_expl, user_based_test_dict_expl)
        print(res)

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