
import numpy as np

"""
    Calculates the correlation coefficent
        between two vectors
    
    a, b vectors
    esplicit -> boolean value that  defines if we
    need to scale or not cause in the case of implicit
    we do not need to scale
"""

calculated_esplicit_vectors = {}
calculated_implicit_vectors = {}

def correlation_coefficent(a,b, esplicit):

    a_scaled = a
    b_scaled = b

    if(len(a) != len(b)):
        raise Exception('a and b must have the same lenght')
    
    if(esplicit):
        a_non_zero_vals = np.count_nonzero(a)
        b_non_zero_vals = np.count_nonzero(b)

        avg_a = np.sum(a)/a_non_zero_vals
        avg_b = np.sum(b)/b_non_zero_vals

        a_scaled = [0]*len(a)
        b_scaled = [0]*len(b)

        for i in range(0,len(a)):
            if(a[i] != 0):
                a_scaled[i] = a[i] - avg_a
            
            if(b[i] != 0):
                b_scaled[i] = b[i] - avg_b

    a_scaled_norm = np.linalg.norm(a_scaled)
    b_scaled_norm = np.linalg.norm(b_scaled)

    sim = np.inner(a_scaled,b_scaled)/(a_scaled_norm*b_scaled_norm)
    #print(sim)
    return sim

"""
    Returns the similarity between x 
    and the other users 
    
    x -> userID
    d -> matrix of all the users
        rows -> users
        cols -> items
    
    esplicit -> boolean value that  defines if we
    need to scale or not cause in the case of implicit
    we do not need to scale

"""
def most_similar_users (x,d, esplicit):

    x_vect = d[x]

    vect_sim = []

    for user in range(0, len(d)):
        if(x == user):
            continue
        
        sim_x_user = correlation_coefficent(x_vect, d[user], esplicit)
        vect_sim.append([sim_x_user, user])
    
    vect_sim = np.array(vect_sim)
    
    return vect_sim

"""
    This is not a normal sum of vectors
    cause each vector v1, v2 has each component
    as follows: [sim_score, user_id]
"""
def sum_vectors(v1, v2, a, b):
    ret = []
    for i in range(0, len(v1)):
        pass
        if(v1[i][1] != v2[i][1]):
            raise Exception('The vector has to have the same users in the same positions')
        summed_sim = v1[i][0]*a + v2[i][0]*b
        ret.append([summed_sim, v1[i][1]])
    
    return ret
        

"""
    Returns the prediction of the rating
    that the user x cuold give to item i
1
    x -> userID
    esplicit -> esplicit matrix of all the users
        rows -> users
        cols -> items
    implicit -> implicit data
    i -> itemID
    k -> number of users to return
    a and b coefficent to weight 
    esplicit and implicit ratings 
"""
def prediction(x,esplicit,implicit,i,k,a,b):
    if(calculated_esplicit_vectors.get(x) == None):
        calculated_esplicit_vectors[x] = most_similar_users(x,esplicit, True)
        calculated_implicit_vectors[x] = most_similar_users(x,implicit, False)

    esplicit_implicti_sim = sum_vectors(calculated_esplicit_vectors[x],
                                        calculated_implicit_vectors[x],
                                        a,b)

    esplicit_implicti_sim = np.array(esplicit_implicti_sim) 

    sorted = np.flip(esplicit_implicti_sim[np.argsort(esplicit_implicti_sim[:,0])])   

    print(sorted)
   
    denominator = 0
    numerator = 0
    counter = 0
    for index in range(0, len(sorted)):

        if(counter == k):
            break
        user = int(sorted[index][0])
        
        sim_user_x = sorted[index][1]
        rate_user_i = esplicit[user][i]
        if(rate_user_i != 0):
            counter += 1
            
            denominator += sim_user_x
            numerator += sim_user_x * rate_user_i
    
    prediction = numerator/denominator
    
    print(prediction)

    return prediction

if __name__ == "__main__":
    
    matrix_esplicit = [
        [1,2,3,0],
        [1,0,3,1],
        [0,0,1,3],
        [0,1,5,2],
        [0,1,4,2],
        [1,4,2,5],
        [1,2,0,0]
    ]

    matrix_implicit = [
        [1,0,0,0],
        [1,0,1,1],
        [0,0,0,1],
        [0,1,0,1],
        [0,1,1,0],
        [1,0,1,1],
        [1,1,0,1]
    ]

    prediction(4, matrix_esplicit, matrix_implicit, 2, 2, 0.9,0.1)
    



