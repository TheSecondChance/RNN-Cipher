import numpy as np
from Crypto.Random import get_random_bytes
import csv

def create(block_size, length):
    x_train = np.array(list(get_random_bytes(length * block_size)))
    y_train = np.array(list(get_random_bytes(length * block_size)))

    x_train = np.reshape(x_train, (length, block_size)) / 256
    y_train = np.reshape(y_train, (length, block_size)) / 256

    return (x_train, y_train)

def save(dataset, path_to_file):
    x_train, y_train = dataset
    with open(path_to_file, 'w', newline='') as f:
        writer = csv.writer(f)       
        
        for x in x_train:
            writer.writerow(x)      
        f.write('\n')      
        for y in y_train:
            writer.writerow(y)
    
def load(path_to_file):
    x_train = []
    y_train = []
    
    with open(path_to_file, newline='') as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                break
            x_train.append([float(i) for i in row])
        for row in reader:
            y_train.append([float(i) for i in row])
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return (x_train, y_train)