
import load_files as load
import multiprocessing
import numpy as np
import math

calculated_esplicit_dict = {}
calculated_implicit_dict = {}

cliques = {}

CORES = multiprocessing.cpu_count()

"""
    Calculates the average value of the 
    valeues in a

    a -> dictionary of non zero values 
"""
def avg(a):
    i=0
    tot = 0
    for k in a.keys():
        i +=1
        tot += a.get(k)
    
    return tot/i

"""
    Calculates a new dictionary that for each
    non null value of a has that values - const

    returns a - const

    a -> dictioary of non zero values
"""
def scale(a,const):
    ret = {}
    
    for k in a.keys():
        pass
        ret[k] = a.get(k) - const
    
    return ret

"""
    calculates the norm of a dictionary

    a -> dictioary of non zero values

"""
def norm (a):
    tot = 0
    for k in a.keys():
        tot += pow(a.get(k),2) 

    return math.sqrt(tot)

"""
    computes the inner product of two dictionaries

    a -> dictioary of non zero values
    b -> dictioary of non zero values

"""
def inner_product(a,b):
    tot = 0
    for k in a.keys():
        b_tmp = b.get(k)
        if(b_tmp != None):
            tot += b_tmp * a.get(k) 

    return tot

"""
    Calculates the correlation coefficent
        between two vectors
    
    a, b dictionaries of non zero values

    esplicit -> boolean value that  defines if we
    need to scale or not cause in the case of implicit
    we do not need to scale
"""
def compute_correlation_coefficent(a,b, esplicit):

    a_scaled = a
    b_scaled = b

    if(esplicit):

        avg_a = avg(a)
        avg_b = avg(b)

        a_scaled = scale(a, avg_a)
        b_scaled = scale(b, avg_b)

    a_scaled_norm = norm(a_scaled)
    b_scaled_norm = norm(b_scaled)

    if(a_scaled_norm == 0 or b_scaled_norm == 0):
        return 0

    sim = inner_product(a_scaled,b_scaled)/(a_scaled_norm*b_scaled_norm)

    return sim

import correlation_coefficent_thread 


"""
    returns a list containing num_folds lists 
    that are a partition of data

    data -> list
    num_folds -> number of partitions 
"""
def make_partitions(num_folds, data):
    data_size = len(data)
    fold_size = data_size // num_folds
    partitions = []
    for idx in range(num_folds-1):
        partitions.append(data[idx*fold_size:(idx+1)*fold_size])
    partitions.append(data[idx*fold_size:])
    return partitions

"""
    Returns the clique of the user as np.array

    user -> userID
    utility_matrix -> {user:{item:rating}}
    clique_size -> size of the clique of the user
"""
import correlation_coefficent_thread 
def compute_clique_without_implicit(user, utility_matrix, clique_size):

    unique_users = list(set(utility_matrix.keys()))
    user_dict = utility_matrix[user]
    partitions = make_partitions(CORES,unique_users)    
        

    threads = []
    for idx in range(CORES):
        name = "Thread-{}".format(idx)
        thread = correlation_coefficent_thread.ComputeCorrelationCoefficentThread(name = name, 
                                                user_dict = user_dict, 
                                                user_ids = partitions[idx], 
                                                utility_matrix = utility_matrix, 
                                                is_explicit = True)
        threads.append(thread)

    
    [thread.start() for thread in threads]    
    print('======= all threads started =======')

    results = [thread.join() for thread in threads]
    print('======= all threads joined =======')
    
    similarities = []
    for elem in results:
        for similarity in elem:
            similarities.append(similarity)
    
    similarities = np.array(similarities)
    clique = similarities[np.argsort(similarities[:,1])[::-1]][1:clique_size+1]

    return clique

"""
    weighted sum of scores of v1 and v2    

    v1 -> list [[ID, score]]
    v2 -> list [[ID, score]] 

    a, b weights
"""
def sum_vectors(v1, v2, a, b):
    ret = []

    for idx in range(len(v1)):
        if(v1[idx][0]!=v2[idx][0]):
            print('The two vectors has to have the same order of users!\nIf you see this message than you should change the return of ComputeCorrelationCoefficentThread.join() into a dictionary so that I can access the implicit and explicit scores for the same user with a small cost')
            raise Exception('The two vectors must have the same order of users!')
        
        summed_sim = v1[idx][1]*a + v2[idx][1]*b
        ret.append([v1[idx][0], summed_sim])
        
    print(ret)
    return ret

def compute_clique_with_implicit(user, esplicit_utility_matrix, implicit_utility_matrix, clique_size, a, b):

    unique_users = list(set(esplicit_utility_matrix.keys()))
    user_dict = esplicit_utility_matrix[user]
    partitions = make_partitions(CORES,unique_users)    
        
    print('======= start creating esplicit threads =======')

    threads = []
    for idx in range(CORES):
        name = "Thread-{}".format(idx)
        thread = correlation_coefficent_thread.ComputeCorrelationCoefficentThread(name = name, 
                                                user_dict = user_dict, 
                                                user_ids = partitions[idx], 
                                                utility_matrix = esplicit_utility_matrix,
                                                is_explicit = True)
        threads.append(thread)
    
    [thread.start() for thread in threads]    
    print('======= all esplicit threads started =======')

    results = [thread.join() for thread in threads]
    print('======= all esplicit threads joined =======')
    
    esplicit_similarities = []
    for elem in results:
        for similarity in elem:
            esplicit_similarities.append(similarity)
    
    print('======= start creating implicit threads =======')

    user_dict = implicit_utility_matrix[user]
    threads = []
    for idx in range(CORES):
        name = "Thread-{}".format(idx+CORES)
        thread = correlation_coefficent_thread.ComputeCorrelationCoefficentThread(name = name, 
                                                user_dict = user_dict, 
                                                user_ids = partitions[idx], 
                                                utility_matrix = implicit_utility_matrix,
                                                is_explicit = False)
        threads.append(thread)
    
    [thread.start() for thread in threads]    
    print('======= all implicit threads started =======')

    results = [thread.join() for thread in threads]
    print('======= all implicit threads joined =======')
    
    implicit_similarities = []
    for elem in results:
        for similarity in elem:
            implicit_similarities.append(similarity)

    similarities = sum_vectors(esplicit_similarities, implicit_similarities, a, b)

    similarities = np.array(similarities)
    clique = similarities[np.argsort(similarities[:,1])[::-1]][1:clique_size+1]

    return clique



if __name__ == "__main__":
    
    """ dict_esplicit = {
        0 : {0:1, 1:2, 2:3     },
        1 : {0:1,      2:3, 3:1}, 
        2 : {          2:1, 3:3},   
        3 : {     1:1, 2:5, 3:2},   
        4 : {     1:1, 2:4, 3:2},   
        5 : {0:1, 1:4, 2:2, 3:5},   
        6 : {0:1, 1:2          }
    }

    dict_implicit = {
        0 : {0:1               },
        1 : {0:1,      2:1, 3:1}, 
        2 : {               3:1},   
        3 : {     1:1,      3:1},   
        4 : {     1:1, 2:1,    },   
        5 : {0:1,      2:1, 3:1},   
        6 : {0:1, 1:1,      3:1}
    }

    prediction_with_implicit(4, dict_esplicit, dict_implicit, 2, 2,0.9,0.1)  """
    

    #dict_esplicit = load.s_load('/home/francesco/Desktop/WIR-project/Datasets/movies_dataset/utility_matrix_item_based.txt')

    float_dict = {}

    for k1 in dict_esplicit.keys():
        float_dict[k1] = {}
        for k2 in  dict_esplicit[k1].keys():
            float_dict[k1][k2] = float(dict_esplicit[k1][k2])
            

    k = list(dict_esplicit.keys())[0]
    k2 = list(dict_esplicit[k].keys())[0]
    
    




