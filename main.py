from load_files import divide_dict_into_k
from load_files import merge_dicts
from load_files import s_load
import correlation_coefficent 
import numpy as np
import prediction_thread
import multiprocessing
CORES = multiprocessing.cpu_count()

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
    returns the K items with higher score for 
    the user 
"""
def top_k_without_implicit(user, dict_esplicit, set_of_items, k):
    float_dict = {}
    predicted = []
    for k1 in dict_esplicit.keys():
        float_dict[k1] = {}
        for k2 in  dict_esplicit[k1].keys():
            float_dict[k1][k2] = float(dict_esplicit[k1][k2])
    
    items = list(set_of_items.difference(set(dict_esplicit[user].keys())))

    predicted.append([correlation_coefficent.prediction_without_implicit(user,float_dict,items[0],10), items[0]])

    user_index = 1
    while(user_index < len(items)):
        threads = []
        
        for i in range(0,CORES):
            
                
            if(user_index + i == len(items)):
                break
            
            th = prediction_thread.PredictionThread(user, dict_esplicit, {}, False, items[user_index + i], k, 0, 0)
            th.start()
            threads.append(th)
        
        for th in threads:
            th.join()
            res_i = th.get_res()
            predicted.append([res_i, th.get_item()]) 

        user_index += CORES
    predicted = np.array(predicted) 

    sorted_predictions = np.flip(predicted[np.argsort(predicted[:,0])])

    print('sorted_predictions->')
    print(sorted_predictions)

    return sorted_predictions[:10]

from datetime import datetime
if __name__ == "__main__":
    dict_esplicit = s_load('/home/francesco/Desktop/WIR-project/Datasets/movies_dataset/utility_matrix_user_based2.txt')
    item_dict_esplicit = s_load('/home/francesco/Desktop/WIR-project/Datasets/movies_dataset/utility_matrix_item_based.txt')

    items = set(item_dict_esplicit.keys())
    dicts = divide_dict_into_k(dict_esplicit, 4)
    
    d0 = dicts[0]
    d1 = dicts[1]
    d2 = dicts[2]
    d3 = dicts[3]

    user = list(d0.keys())[0]    
    print('user:{}'.format(user))

    d_123 = merge_dicts([d1,d2,d3])
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    
    res = top_k_without_implicit(user, d_123, items, 100)

    print(res)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

