import numpy as np
import json
import random

def map_choices(narray):
    map_file = {}
    choices = [i for i in range(len(narray))]
    for i in range(len(narray)):
        r = random.choice(choices)
        #print(r)
        map_file[i] = r
        choices.remove(r)
    return map_file
    
def encrypt_order(file, map_file):
    new_file = [None]*len(file)
    for i in map_file:
        index_file = int(i)
        index_new_file = int(map_file[i])
        new_file[index_new_file] = file[index_file]
    return new_file
    
def encrypt_file(file_path):
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    file = np.load(file_path)
    np.load = np_load_old
    
    mapping_layer = map_choices(file)
    encrypted_file = encrypt_order(file, mapping_layer)
    np.save('encrypted_'+file_path, encrypted_file)
    with open('mapping_layer', 'w+') as w:
        w.write(json.dumps(mapping_layer))
        
    return 'mapping_layer', 'encrypted_'+file_path
    
def decrypt_order(file, map_):
    new_file = [None]*len(file)
    #print(map_)
    for i in map_:
        index_file = int(map_[i])
        index_new_file = int(i)
        new_file[index_new_file] = file[index_file]
    return new_file
    
def decrypt_file(file_path, mapping_layer):
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    file = np.load(file_path)
    np.load = np_load_old
    with open(mapping_layer, 'r') as w:
        mapping_layer = json.load(w)
    decrypted_file = decrypt_order(file, mapping_layer)
    np.save(file_path, decrypted_file)