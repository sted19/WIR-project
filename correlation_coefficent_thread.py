from correlation_coefficent import correlation_coefficent
import threading

class CorrelationCoefficentThread (threading.Thread):
   def __init__(self, x_dict, user_dict, user, esplicit):
      threading.Thread.__init__(self)
      self.x_dict = x_dict
      self.user_dict = user_dict
      self.esplicit = esplicit
      self.user = user
      self.result = None
   
   def run(self):
      res = correlation_coefficent(self.x_dict, self.user_dict, self.esplicit)
      self.result = [self.user, res]

   """
      returns a vector with two elements
      in first position the user 
      in the second position the similarity 
      between x and user
   """
   def get_res(self):
      return self.result