from random import randint
import json


def add_element(d, user_id, item_name, rating, item_based):
    if(item_based):
        if(d.get(item_name) == None):
            d[item_name] = {}
    
        d[item_name][user_id] = rating
    else:
        if(d.get(user_id) == None):
            d[user_id] = {}
        
        d[user_id][item_name] = rating
"""
    Load data stored by Lorenzo
"""
def l_load(file_name, item_based):

    implicit = {}
    esplicit = {}
    
    f = open(file_name, 'r')
    fields = f.readline().strip().split('\t') # first line has no data

    item_name_index = fields.index('bookTitle')+1
    user_id_index = fields.index('userId')+1
    rating_index = fields.index('bookRating')+1

    while(True):
        line = f.readline().strip().split('\t')
        if(line == ['']):
            break

        item_name = line[item_name_index]
        user_id = line [user_id_index]
        rating = float(line[rating_index])
        
        if(rating == 0): # implicit rating
            add_element(implicit, user_id, item_name, 1, item_based)
            
        else:
            add_element(esplicit, user_id, item_name, rating, item_based)
          
    
    return (implicit, esplicit)

"""
    Load data stored by stefano
"""
def s_load(file_name):
    with  open(file_name) as fr:
        data = json.load(fr)
        return data

"""
    Returns a vector of k dictionaries
    that are a partition of d

    d -> dictionary
    k -> integer
"""
def divide_dict_into_k(d, k):
    dicts = [{}]*k

    for key1 in d.keys():
        for key2 in d[key1].keys():
            rand_index = randint(0,k-1)
            
            tmp_dict = dicts[rand_index]

            if(tmp_dict.get(key1) == None):
                tmp_dict[key1] = {}
            
            tmp_dict[key1][key2] = d[key1][key2]

    return dicts       

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