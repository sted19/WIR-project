from correlation_coefficent import compute_correlation_coefficent
import threading
class ComputeCorrelationCoefficentThread (threading.Thread):
   """
      user_dict -> dictionary of the user: {itemID:rating}
      user_ids -> set of users IDs
      utility_matrix -> dataset
      is_explicit -> boolean value that  defines if we
         need to scale or not cause in the case of implicit
         we do not need to scale
   """
   def __init__(self, user_dict, user_ids, utility_matrix, is_explicit, name = None):
      threading.Thread.__init__(self)
      self.name = name
      self.user_dict = user_dict
      self.user_ids = user_ids
      self.is_explicit = is_explicit
      self.utility_matrix = utility_matrix
      self.result = None
   
   def run(self):
      if self.name == None:
         raise Exception('Something bad happened, thread has no name')
      print('Thread {} started'.format(self.name))
      self.compute_similarities()

   
   def compute_similarities(self):
      similarities = []
      for user in self.user_ids:
         tmp_user_dict = self.utility_matrix[user]
         similarity = compute_correlation_coefficent(self.user_dict, tmp_user_dict, self.is_explicit)
         similarities.append([user,similarity])
      
      self.result = similarities

   # vector of similarities --> [userID, sim_score]
   def join(self):
        threading.Thread.join(self)
        if self.result is not None:
            return self.result
        else:
            print('Error using threads, result is None')