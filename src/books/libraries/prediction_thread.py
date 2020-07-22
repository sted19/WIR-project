from correlation_coefficent import prediction_without_implicit
from correlation_coefficent import prediction_with_implicit
import threading

# 

class PredictionThread (threading.Thread):
    def __init__(self, user_id, explicit, implicit, implicit_bool, i, k, a, b):
        threading.Thread.__init__(self)
        self.implicit = implicit 
        self.explicit = explicit
        self.user_id = user_id
        self.i = i
        self.k = k
        self.a = a
        self.b = b
        self.implicit_bool = implicit_bool
        self.result = None
   
    def run(self):
        if(self.implicit_bool):
            
            self.result = prediction_with_implicit(self.user_id, self.explicit, 
                                        self.implicit, self.i, self.k, self.a, self.b)
        else:
            self.result = prediction_without_implicit(self.user_id, self.explicit, 
                                                         self.i, self.k)

    """
        returns a vector with two elements
        in first position the user 
        in the second position the similarity 
        between x and user
    """
    def get_res(self):
        return self.result
    def get_item(self):
        return self.i