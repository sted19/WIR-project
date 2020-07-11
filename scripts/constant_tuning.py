from load_files import divide_dataset
from load_files import merge_dicts
from load_files import l_load
from datetime   import datetime
from main       import predict

import correlation_coefficent 
import multiprocessing
import numpy as np
import os

from constants import *

def constant_tuning(train_dict_expl, train_dict_impl):

    set_of_users_expl = set(train_dict_expl.keys()) # I am considering for the clique only the users 
                                                    # in the explicit data, not the ones present only in the implicit one
    print(len(set_of_users_expl))
    folds_explicit = divide_dataset(train_dict_expl, num_folds)

    errors = {}

    users = set(train_dict_expl.keys())
    print(len(users))
    
    for i in range(num_folds):
        users = users.intersection(set(folds_explicit[i].keys()))

    users = users.intersection(set(train_dict_expl.keys()))
    print(len(users))

    for test_fold_idx in range(num_folds):
        train_dict_explicit = merge_dicts(folds_explicit[:test_fold_idx] + folds_explicit[test_fold_idx+1:])
        test_dict = folds_explicit[test_fold_idx] 

        train_dict_implicit = train_dict_impl 

        new_test_fold = True

        for a_numerator in range(1,20):

            print('a: {}'.format(a_numerator))
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            
            a = a_numerator/20
            tot_error = 0


            for user in users:    
                clique = correlation_coefficent.compute_clique_with_implicit(user, train_dict_explicit, train_dict_implicit, 100, a, 1-a, users = set_of_users_expl,  new_test_fold=new_test_fold)
                test_items = list(test_dict[user].keys())

                for item in test_items:
                
                    prediction = predict(user, item, clique, train_dict_explicit)
                    tmp_error = abs(prediction - test_dict[user][item])

                    tot_error += tmp_error 

            new_test_fold = False

            if(errors.get(a) == None):
                errors[a] = 0
            errors[a] += tot_error
    
    print('errors ->')
    print(errors)

if __name__ == "__main__":
    pass

