# Minh Nguyen - minh.nguyen@hunter.cuny.edu
# Machine Learning - Decision Tree.

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Function to split the dataset into smaller pieces with size = 10%, 20%, 50%, 80%, and 100% of original dataset for learning curve.
def split_dataset(training_data, dsize):

        # Last column, i.e. labels.
        features_train = training_data.values[:,0:64]
        # The rest, i.e. features.
        labels_train = training_data.values[:,-1]
  
        # Split the training data, in this case we only need 'features_train2' and 'labels_train2'.
        features_train1, features_train2, labels_train1, labels_train2 = train_test_split(features_train, labels_train, test_size = dsize)
	# E.g. If test_size = 0.2 meaing we'll take 20% of the dataset and ignore 80% of the rest.
      
        return features_train, labels_train, features_train1, features_train2, labels_train1, labels_train2

# Funtion to grow a decision tree with training data and entropy. 
def grow_dt(features_train, labels_train, maximum_depth):

        # Create Decision Tree classifer object.
        clf = DecisionTreeClassifier(criterion = "entropy", max_depth = maximum_depth, min_samples_split = 2, min_samples_leaf = 1)
  
        # Train Decision Tree Classifer.
        clf.fit(features_train, labels_train)

        return clf

# Function to test the decision tree on testing data.
def test(features_test, clf): 
  
        # Predict the labels of testing data based on the training data Decision Tree classifier
        labels_pred = clf.predict(features_test)

        return labels_pred

# Function to calculate the impact of maximum tree depth.
def dt_max_depth(training_data, testing_data):

        # Last column, i.e. labels.
        features_train = training_data.values[:,0:64]
        # The rest, i.e. features.
        labels_train = training_data.values[:,-1]
        features_test = testing_data.values[:,0:64]
        labels_test = testing_data.values[:,-1]

        maximum_depth = np.array([0, 3, 4, 6, 12, 14]) # Maximum depth array.
        acc = np.array([0, 0, 0, 0, 0, 0]) # Accuracy array, respectively.

        print ()
        print ("Maximum Depth: Below is the impact of different maximum depth (3, 6, and 14 respectively) for accuracy, using the entire training data. For the rest, please see Figure 1.")
        print ()

        # Loop thru each case of the maximum depth  
        for s in range(1, 6):

              clf = grow_dt(features_train, labels_train, maximum_depth[s])

              labels_pred = test(features_test, clf)

              all_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
              count = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
              true_count = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

              for i in range(0, 10):
               for j in range(0, len(labels_pred)):
                 if all_labels[i] == labels_pred[j]:
                      count[i] =  count[i]+1
                      if labels_pred[j] == labels_test[j]:
                             true_count[i] = true_count[i]+1
              
              test_cases = 0
              no_err = 0
              for i in range(0, 10):
                     test_cases = test_cases + count[i]
                     no_err = no_err + true_count[i]
              acc[s] = (no_err/test_cases)*100
              
              if (s == 5 or s == 3 or s == 1):
                      print ("Maximum depth:", maximum_depth[s])
                      print ("     ", "Test Cases", "      ", "True", "  ", "False")
                      for i in range(0, 10):
                                print (all_labels[i], "	", count[i], "		", true_count[i], "	", count[i]-true_count[i])
                      print ("T", "	", test_cases, "		", no_err, "	", test_cases-no_err)
                      print ("Accuracy: ", round((no_err/test_cases)*100, 2))
                      print ("Error rate: ", round(((test_cases-no_err)/test_cases)*100, 2))
                      print ()

        print ("Maximum Depth: Above is the impact of different maximum depth (3, 6, and 14 respectively) for accuracy, using the entire training data. For the rest, please see Figure 1.")
        print ()
        plt.figure(1)
        plt.plot(acc, maximum_depth, 'ro', acc, maximum_depth, 'k')
        plt.axis([0, 100, 0, 15])
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Maximum depth")
        plt.title("Impact of maximum depth.")

# Function to calculate the learning curve given different sizes of training data.
def learning_curve(training_data, testing_data, maximum_depth):

        # Last column, i.e. labels.
        features_train = training_data.values[:,0:64]
        # The rest, i.e. features.
        labels_train = training_data.values[:,-1]
        features_test = testing_data.values[:,0:64]
        labels_test = testing_data.values[:,-1]

        dsize = np.array([0, 10, 20, 50, 80, 99.99]) # 0%, 10%, 20%, 50%, 80%, and 100% of the dataset.
        acc = np.array([0, 0, 0, 0, 0, 0]) # Accuracy array, respectively.
        number_of_examples = np.array([0, 0, 0, 0, 0, 0]) # Based on dsize and 'labels_test'.

        for s in range(1, 6):
               number_of_examples[s] = (dsize[s]/100)*len(labels_train)

        print ("Learning Curve: Below is the comparison when we use 10%, 50%, and 100% of the training data respectively, s.t. maximum depth of", maximum_depth, ". For the rest, please see Figure 2.")
        print ()

        # Loop thru each case of the dataset   
        for s in range(1, 6):
              features_train, labels_train, features_train1, features_train2, labels_train1, labels_train2 = split_dataset(training_data, dsize[s]/100)

              clf = grow_dt(features_train2, labels_train2, maximum_depth)

              labels_pred = test(features_test, clf)

              all_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
              count = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
              true_count = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

              for i in range(0, 10):
               for j in range(0, len(labels_pred)):
                 if all_labels[i] == labels_pred[j]:
                      count[i] =  count[i]+1
                      if labels_pred[j] == labels_test[j]:
                             true_count[i] = true_count[i]+1
              
              test_cases = 0
              no_err = 0
              for i in range(0, 10):
                     test_cases = test_cases + count[i]
                     no_err = no_err + true_count[i]
              acc[s] = (no_err/test_cases)*100
              
              if (s == 5 or s == 1 or s == 3):
                 if (s == 5):
                      print (dsize[s]+0.01, "% of the training data, number of examples:", number_of_examples[s]+1)
                      print ("     ", "Test Cases", "      ", "True", "  ", "False")
                      for i in range(0, 10):
                                print (all_labels[i], "	", count[i], "		", true_count[i], "	", count[i]-true_count[i])
                      print ("T", "	", test_cases, "		", no_err, "	", test_cases-no_err)
                      print ("Accuracy: ", round((no_err/test_cases)*100, 2))
                      print ("Error rate: ", round(((test_cases-no_err)/test_cases)*100, 2))
                      print ()
                 else:
                      print (dsize[s], "% of the training data, number of examples:", number_of_examples[s])
                      print ("     ", "Test Cases", "      ", "True", "  ", "False")
                      for i in range(0, 10):
                                print (all_labels[i], "	", count[i], "		", true_count[i], "	", count[i]-true_count[i])
                      print ("T", "	", test_cases, "		", no_err, "	", test_cases-no_err)
                      print ("Accuracy: ", round((no_err/test_cases)*100, 2))
                      print ("Error rate: ", round(((test_cases-no_err)/test_cases)*100, 2))
                      print ()

        print ("Note 1: Learning curve, above is the comparison when we use 10%, 50%, and 100% of the training data respectively, s.t. maximum depth of", maximum_depth, ". For the rest, please see Figure 2.")
        print ("Note 2: Please scroll up to see the impact of maximum depth for the accuracy.")
        print ()

        plt.figure(2)
        plt.plot(acc, number_of_examples, 'ro', acc, number_of_examples, 'k')
        plt.axis([0, 100, 0, 4000])
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Number of examples")
        plt.title("Learning curve given different number of examples.")

def main():       

        testing_data = pd.read_csv('optdigits.tes', sep = ',', header = None)
        training_data = pd.read_csv('optdigits.tra', sep = ',', header = None)

        # Last column, i.e. labels.
        features_train = training_data.values[:,0:64]
        # The rest, i.e. features.
        labels_train = training_data.values[:,-1] 
        features_test = testing_data.values[:,0:64]
        labels_test = testing_data.values[:,-1]       

        dt_max_depth(training_data, testing_data)
        learning_curve(training_data, testing_data, 20)
        plt.show()

if __name__ == "__main__":
        main()
