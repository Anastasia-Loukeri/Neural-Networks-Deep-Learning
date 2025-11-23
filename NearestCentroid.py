import numpy as np
import pickle
import os


CIFAR10_DIR =  'C:/Users/User/OneDrive/Υπολογιστής/CIFAR_KNN/cifar-10-batches-py'

def load_pickle_batch(filename):
    """Load a single CIFAR-10 batch from a pickle file."""
    with open(filename, 'rb') as f:
        # Use 'latin1' encoding for Python 3 compatibility with Python 2 pickled data
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        # Αναμόρφωση δεδομένων εικόνας
        
        return X, np.array(Y)

def load_cifar10(root_dir):
    
    
    # Φόρτωση των 5 training batches
    X_train_batches, y_train_batches = [], []
    for i in range(1, 6):
        fpath = os.path.join(root_dir, 'data_batch_%d' % (i, ))
        X, Y = load_pickle_batch(fpath)
        X_train_batches.append(X)
        y_train_batches.append(Y)    
    
    # Concatenate all training batches
    X_train = np.concatenate(X_train_batches)
    y_train = np.concatenate(y_train_batches)
    
    # Φόρτωσε το test batch
    fpath = os.path.join(root_dir, 'test_batch')
    X_test, y_test = load_pickle_batch(fpath)
    
    # Data flattened (N, 3072) from pickle 
    # Κανονικοποίηση των πίξελ
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    return X_train, y_train, X_test, y_test

# Load the data using the custom function
print("Φόρτωση δεδομένων από pickle batches...")
X_train_flat, y_train, X_test_flat, y_test = load_cifar10(CIFAR10_DIR)

print(f"Moρφή των τrain data : {X_train_flat.shape}")
print(f"Μορφή των Test data : {X_test_flat.shape}")


class NearestCentroidFromScratch:
    
    def __init__(self):
        self.centroids = {}
        self.classes = None

    def fit(self, X, y):
        """Calculates the centroid (mean) for each class."""
        self.classes = np.unique(y)
        
        for c in self.classes:
            # Select all samples belonging to the current class 'c'
            X_c = X[y == c]
            # Compute the centroid: the mean of all samples in the class
            self.centroids[c] = np.mean(X_c, axis=0)
            
        print(f"Υπολογισμός των κεντρών για  {len(self.classes)} κλάσεις.")

    def predict(self, X):
        """Predicts the class label for each sample in X."""
        predictions = []
        
        # Iterate over each test sample
        for x_sample in X:
            distances = {}
            
            # Calculate the Euclidean distance (squared L2 norm) to every centroid
            for c, centroid in self.centroids.items():
                distance_sq = np.sum((x_sample - centroid) ** 2)
                distances[c] = distance_sq
            
            # Find the class with the minimum distance (the nearest centroid)
            nearest_class = min(distances, key=distances.get)
            predictions.append(nearest_class)
            
        return np.array(predictions)

# Initialize and train the classifier
nc_model = NearestCentroidFromScratch()
print("Training Nearest Centroid Classifier...")
nc_model.fit(X_train_flat, y_train)

def main():
    """
    The main function to execute the Nearest Centroid classification pipeline.
    """
    
    # 1. Load Data
    
    try:
        X_train_flat, y_train, X_test_flat, y_test = load_cifar10(CIFAR10_DIR)
        print(f"Φόρτωση δεδομένων: Train {X_train_flat.shape}, Test {X_test_flat.shape}")
    except FileNotFoundError:
        print(f"Δεν βρέθηκε το CIFAR-10 data directory στο path '{CIFAR10_DIR}'.")
        
        return # Exit the function if data isn't found

    # 2. Initialize and Train the Classifier
    nc_model = NearestCentroidFromScratch()
    
    nc_model.fit(X_train_flat, y_train)

    # 3. Predict and Evaluate
    
    y_pred = nc_model.predict(X_test_flat)

    # Calculate Accuracy
    correct_predictions = np.sum(y_pred == y_test)
    total_samples = len(y_test)
    accuracy = correct_predictions / total_samples

    # 4. Report Results
    print("\n" + "="*40)
    print(f"Τελικά αποτελέσματα για πλήθος test samples: {total_samples}")
    print(f"Σωστά ταξινομημένα: {correct_predictions}")
    print(f"Ακρίβεια (Accuracy): {accuracy*100:.2f}%")
    print("="*40)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    # 2. Υπολογισμός της Confusion Matrix (scikit-learn)
    cm = confusion_matrix(y_test, y_pred)
    
    # 3. Απεικόνιση (Visualization)
    print("\n--- Confusion Matrix ---")
    plt.figure(figsize=(10, 8))
    
    # Χρήση του Seaborn για καλύτερη εμφάνιση
    sns.heatmap(
        cm, 
        annot=True,        # Εμφάνιση των αριθμών στα κελιά
        fmt='d',           # Μορφοποίηση ως ακέραιος
        cmap='Blues',      # Χρωματικό σχήμα
        xticklabels=classes, 
        yticklabels=classes
    )
    
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual  Class')
    plt.title(f'Confusion Matrix for k={k_value}')
    plt.show()


if __name__ == '__main__':
    main()