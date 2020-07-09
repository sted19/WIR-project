from random import randint
import json


def add_element(d, user_id, item_id, rating, item_based):
    
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
    folds = [{}]*num_folds

    for key1 in dataset.keys():
        for key2 in dataset[key1].keys():
            rand_index = randint(0,num_folds-1)
            
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
    
from datetime import datetime
if __name__ == "__main__":
    #(implicit, esplicit) = l_load('/home/francesco/Desktop/WIR-project/Datasets/BX-CSV-Dump/item_based_implicit.csv', True)
    #print(implicit)

    data = s_load('/home/francesco/Desktop/WIR-project/Datasets/movies_dataset/utility_matrix_user_based2.txt')

    now = datetime.now()


    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    print("start divide")

    dicts = divide_dict_into_k(data, 4)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    print("start merge")

    dicts1 = dicts[:3]

    res = merge_dicts(dicts1)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    print("end 1 merge")

    """ for i in range(0,4):
        json.dump(dicts[i], open('/home/francesco/{}.json'.format(i), 'w')) """

    pass