import numpy as np
import json
from constants import *


def add_element(d, user_id, item_id, rating, item_based):
    
    if(rating == 0): #cause only in the case of implicit values the rating can be 0
        rating = 1 

    user_id_ = ''
    item_id_ = ''

    if('X' == user_id[-1] or 'x' == user_id[-1]):
        user_id_ = user_id[:-1] + '1'
    elif( user_id == 'B00009ANY9'):
        user_id_ = '1'
    else:
        user_id_ = user_id + '0'
    
    if('X' == item_id[-1] or 'x' == item_id[-1]):
        item_id_ = item_id[:-1] + '1'
    elif( item_id == 'B00009ANY9'):
        item_id_ = '1'
    else:
        item_id_ = item_id + '0'

    user_id_ = float(user_id_)
    item_id_ = float(item_id_)

    if(item_based):
        if(d.get(item_id_) == None):
            d[item_id_] = {}

        d[item_id_][user_id_] = float(rating)

    else:
        if(d.get(user_id_) == None):
            d[user_id_] = {}
        
        d[user_id_][item_id_] = float(rating)
"""
    Load data stored by Lorenzo
"""
def l_load(file_name, item_based):

    dictionary = {}
    
    f = open(file_name, 'r')
    fields = f.readline().strip().split('\t') # first line has no data

    item_uisbn_index = fields.index('uniqueISBN')+1
    user_id_index = fields.index('userId')+1
    rating_index = fields.index('bookRating')+1

    while(True):
        line = f.readline().strip().split('\t')
        if(line == ['']):
            break

        item_uisbn = line[item_uisbn_index]
        user_id = line [user_id_index]
        rating = float(line[rating_index])
        
        add_element(dictionary, user_id, item_uisbn, rating, item_based)
          
    
    return dictionary

"""
    Load data stored by stefano
"""
def s_load(file_name):
    with  open(file_name) as fr:
        data = json.load(fr)
        return data

"""
    Returns a partition of dataset
    into num_folds dictionaries

    the return is [{key1:{key2:rating}}]
"""
def divide_dataset(dataset, num_folds):
    folds = []
    for i in range(num_folds):
        folds.append({})

    for key1 in dataset.keys():
        for key2 in dataset[key1].keys():
            rand_index = int(np.random.randint(num_folds))
            tmp_dict = folds[rand_index]

            if(tmp_dict.get(key1) == None):
                tmp_dict[key1] = {}
            
            tmp_dict[key1][key2] = dataset[key1][key2]
    
    return folds    

"""
    Returns a dictionary that is the union
    of the values of the dictionaries in dicts

    dicts -> vector of dictionaries
"""
def merge_dicts(dicts):
    ret = dicts[0]

    for i in range(1,len(dicts)):
        tmp_dict = dicts[i]
        for key1 in tmp_dict.keys():
            for key2 in tmp_dict[key1].keys():

                if(ret.get(key1)==None):
                    ret[key1] = {}
                ret[key1][key2] = tmp_dict[key1][key2]

    return ret

'''
    returns a dictionary {item_id:{bookAuthor, topic, bookTitle}}
'''
def build_books_data(file_name):
    dictionary = {}
    
    f = open(file_name, 'r')
    fields = f.readline().strip().split('\t') # first line has no data

    item_uisbn_index = fields.index('uniqueISBN')+1
    author_index = fields.index('bookAuthor')+1
    topic_index = fields.index('topic')+1
    book_title_index = fields.index('bookTitle')+1

    #userId	ISBN	bookTitle	bookAuthor	yearOfPublication	publisher	bookRating	uniqueISBN	topic

    while(True):
        line = f.readline().strip().split('\t')
        if(line == ['']):
            break

        item_uisbn = line[item_uisbn_index]
        topic = line[topic_index]
        book_title = line[book_title_index]
        author = line[author_index]


        if('X' == item_uisbn[-1] or 'x' == item_uisbn[-1]):
            item_uisbn_ = item_uisbn[:-1] + '1'
        elif( item_uisbn == 'B00009ANY9'):
            item_uisbn_ = '1'
        else:
            item_uisbn_ = item_uisbn + '0'

        item_uisbn_ = float(item_uisbn_)
        
        dictionary[item_uisbn_] = {}
        dictionary[item_uisbn_]['bookAuthor'] = author
        dictionary[item_uisbn_]['topic'] = topic
        dictionary[item_uisbn_]['bookTitle'] = book_title
    
    return dictionary

            
        

from datetime import datetime
from constants import *

if __name__ == "__main__":
    explicit_user_based_utility = l_load(explicit_dict_path_books, False)
    folds = divide_dataset(explicit_user_based_utility, 4)

    for key in folds[0].keys():
        for key2 in folds[0][key].keys():
            if(folds[0][key][key2] != folds[1][key][key2] or folds[0][key][key2] != folds[2][key][key2]) or folds[0][key][key2] != folds[3][key][key2]:
                print('diversi')
