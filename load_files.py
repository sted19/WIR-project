"""
    Load data stored by Lorenzo
"""
def l_load(file_name):
    pass
    f = open(file_name)
    while(True):
        line = f.readline().strip().split('\t')
        print(line)    
        

"""
    Load data stored by stefano
"""
import json
def s_load(file_name):
    with  open(file_name) as fr:
        data = json.loads(json.load(fr))
        return data

if __name__ == "__main__":
    s_load('/home/francesco/Desktop/WIR-project/Datasets/movies_dataset/utility_matrix_user_based.txt')