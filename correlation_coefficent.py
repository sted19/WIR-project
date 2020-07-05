
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
def correlation_coefficent(a,b, esplicit):

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
    Returns the similarity between x 
    and the other users 
    
    x -> userID
    d -> dictionary of all the users
        d[user][item] -> rating from user for that item
        if it exists
    
    esplicit -> boolean value that  defines if we
    need to scale or not cause in the case of implicit
    we do not need to scale

"""
def most_similar_users (x,d, esplicit):

    x_dict = d.get(x)
    if(x_dict == None):
        raise Exception("x has to be a real user")

    dict_sim = {}

    users = list(d.keys()) 

    user_index = 0
    while(user_index < len(users)):
        
        threads = []

        for i in range(0,CORES):
            
            if(user_index + i == len(users)):
                break

            user = users[user_index + i]
            
            if(x == user):
                continue
                
            user_dict = d.get(user)
            if(user_dict == None):
                print("This should never happen: correlation_coefficent.py -> most_similar_users()")
                continue
            
            th = correlation_coefficent_thread.CorrelationCoefficentThread(x_dict, user_dict, user, esplicit)
            th.start()
            threads.append(th)
        
        for i in range(len(threads)):
            threads[i].join()
            res_i = threads[i].get_res()
            dict_sim[res_i[0]] =  res_i[1]            

        user_index += CORES
    
    return dict_sim

"""
    weighted sum of scores 
    d1 and d2 dictionries

    a, b weights
"""
def sum_vectors(d1, d2, a, b):
    ret = []

    users_1 = set(d1.keys())
    users_2 = set(d2.keys())

    users = users_1.union(users_2) 

    for user in users:
        
        summed_sim = d1[user]*a + d2[user]*b
        ret.append([summed_sim, int(user)])
        

    return ret
        

"""
    Returns the prediction of the rating
    that the user x cuold give to item i

    x -> userID
    esplicit -> esplicit dictionary of all the users
        esplicit[user][item] = rating from that user to that item
        if present
    implicit -> implicit data
    i -> itemID
    k -> number of users to return
    a and b coefficent to weight 
    esplicit and implicit ratings 
"""
def prediction_with_implicit(x,esplicit,implicit,i,k,a,b):
    clique = cliques.get(x)
    if(clique is None):
        esplicit_scores = most_similar_users(x,esplicit, True)
        implicit_scores = most_similar_users(x,implicit, False)

        esplicit_implicti_sim = sum_vectors(esplicit_scores,
                                            implicit_scores,
                                            a,b)
        esplicit_implicti_sim = np.array(esplicit_implicti_sim) 

        sorted = (np.flip(esplicit_implicti_sim[np.argsort(esplicit_implicti_sim[:,0])]))[:100]                                       

        cliques[x] = sorted

    return prediction(cliques[x], esplicit, i)




"""
    transforms d into an array with the
    following shape 
        [[sim, user]]
"""
def turn_dictionary_into_array_couples(d):
    ret = []
    users = set(d.keys()) 

    for user in users: 
        ret.append((d[user], int(user)))
        
    return ret


"""
    Returns the prediction of the rating
    that the user x cuold give to item i

    x -> userID
    esplicit -> esplicit dictionary of all the users
        esplicit[user][item] = rating from that user to that item
        if present
    i -> itemID
    k -> number of users to return
"""
def prediction_without_implicit(x,esplicit,i,k):
    clique = cliques.get(x)
    if(clique is None):
        
        scores = most_similar_users(x,esplicit, True)

        sim = turn_dictionary_into_array_couples(scores)
        
        print(type(sim[0][0]))
        print(type(sim[0][1]))

        sim  = np.array(sim) 
        print(type(sim[0][0]))
        print(type(sim[0][1]))

        sim = sim[np.argsort(sim[:,0])]
        print('clique not flipped ->')
        print(sim)
        
        sim = (np.flip(sim))[:100]                                       

        cliques[x] = sim

        print('clique flipped ->')
        print(sim)

    return prediction(cliques[x], esplicit, i)


"""
    reutrns the predicted rating
    from a user (to which is associated 
    the vector of similarities) to item i 

    sim -> vector of similariies wiht the 
        following shape: [[sim, user]]
    esplicit -> esplicit dictionary of all the users
        esplicit[user][item] = rating from that user 
        to that item if present
    implicit -> implicit data
    i -> itemID
    k -> number of users to return
"""
def prediction(clique,esplicit,i): 

    denominator = 0
    numerator = 0
    for index in range(0, len(clique)):

        user = str(int(clique[index][0]))
        
        sim_user_x = float(clique[index][1])
        
        if(sim_user_x <= 0):
            print('break at {}'.format(index))
            break
        rate_user_i = esplicit[user].get(i)
        
        if(rate_user_i != None):   
            print('not None')     
            denominator += sim_user_x
            numerator += sim_user_x * rate_user_i
    
    if denominator == 0:
        return 0    #TODO define what to do in the case in which no 
                    #positive user has given a score to that item
                    #we could also retrieve the avg of the scores
                    #of the user
    prediction = numerator/denominator

    print('non merda')
    return prediction

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
    

    dict_esplicit = load.s_load('/home/francesco/Desktop/WIR-project/Datasets/movies_dataset/utility_matrix_item_based.txt')

    float_dict = {}

    for k1 in dict_esplicit.keys():
        float_dict[k1] = {}
        for k2 in  dict_esplicit[k1].keys():
            float_dict[k1][k2] = float(dict_esplicit[k1][k2])
            

    k = list(dict_esplicit.keys())[0]
    k2 = list(dict_esplicit[k].keys())[0]
    
    #print(k) #user -> 102408       item -> 4031
    #print(k2) #user -> 1371.0      item -> 36526.0
    
    #print(dict_esplicit['4031']['36526.0']) #user -> 3     item -> 2,5

    prediction_without_implicit('4031',float_dict,'36526.0',100)
    prediction_without_implicit('4031',float_dict,'36526.0',200)
    prediction_without_implicit('4031',float_dict,'36526.0',500)




