
import numpy as np
import pickle
import os
import sys
import time
from collections import Counter



num_training = 1000  # Αριθμός δειγμάτων εκπαίδευσης
num_test = 100       # Αριθμός δειγμάτων δοκιμής
current_metric='L2'

data_dir = 'C:/Users/User/OneDrive/Υπολογιστής/CIFAR_KNN/cifar-10-batches-py'
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# --------------------------------------------------------------------------
# 1. Βοηθητικές Συναρτήσεις Φόρτωσης Δεδομένων
# --------------------------------------------------------------------------

def load_cifar_batch(filename):
    """Φορτώνει ένα batch του CIFAR-10 από αρχείο pickle."""
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        # Reshape: N x 3 x 32 x 32 -> N x 32 x 32 x 3
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_cifar10(data_dir):
    
    # 5 batches
    X_train_list = []
    Y_train_list = []
    for i in range(1, 6):
        f = os.path.join(data_dir, 'data_batch_%d' % (i,))
        X, Y = load_cifar_batch(f)
        X_train_list.append(X)
        Y_train_list.append(Y)
    X_train = np.concatenate(X_train_list)
    Y_train = np.concatenate(Y_train_list)

    # Φόρτωση test data
    X_test, Y_test = load_cifar_batch(os.path.join(data_dir, 'test_batch'))

    return X_train, Y_train, X_test, Y_test




class KNearestNeighbor(object):
  

    def __init__(self):
        pass

    def train(self, X, y):
        
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, metric='L2'):
      
        num_test = X.shape[0]
        Y_pred=np.zeros(num_test,dtype=self.y_train.dtype)
        
        print(f"Calculate {num_test}x{self.X_train.shape[0]} αποστάσεων...")
        start_time = time.time()
        
        
        dists = np.sqrt(np.sum(np.square(X[:, np.newaxis, :]  - self.X_train), axis=2))

        end_time = time.time()
        print(f"Distances Calculation lasted {end_time - start_time:.2f} secs.")

        for i in range(num_test):
            k_closest_indices=np.argsort(dists[i,:])[:k]
            k_nearest_labels=self.y_train[k_closest_indices]
            Y_pred[i]=Counter(k_nearest_labels).most_common(1)[0][0]

        return Y_pred
    
   



if __name__ == '__main__':
    print("--- Φόρτωση Δεδομένων CIFAR-10 ---")
    try:
        X_full_train, Y_full_train, X_full_test, Y_full_test = load_cifar10(data_dir)
    except FileNotFoundError:
        print(f"File error '{data_dir}' ")
        sys.exit()

    X_train = X_full_train[:num_training]
    Y_train = Y_full_train[:num_training]

    X_test = X_full_test[:num_test]
    Y_test = Y_full_test[:num_test]



    #  Flatten (N, 3072)
    X_train_flat = X_train.reshape(num_training, -1)
    X_test_flat = X_test.reshape(num_test, -1)
    
    print(f"X_train (Flatten): {X_train_flat.shape}")


    classifier = KNearestNeighbor()
    classifier.train(X_train_flat, Y_train)

   
    k_value = 1
    print(f"\n k-NN with k={k_value}")
    
    Y_pred = classifier.predict(X_test_flat, k=k_value, metric='L2')

    # accuracy
    num_correct = np.sum(Y_pred == Y_test)
    accuracy = float(num_correct) / num_test

 
    print(f"Results for {num_training}/{num_test} δείγματα:")
    print(f"  Correctly Classified: {num_correct} / {num_test}")
    print(f"  Accuracy : {accuracy * 100:.2f} %")
  
    

    
