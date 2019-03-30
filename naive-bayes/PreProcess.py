# Minh Nguyen
# minh.nguyen@hunter.cuny.edu

import os
import re

# This class is to generate a 'Bag of Words' (BoW).
class bow(object):
    
    # Initialization.
    def __init__(self):
        self.__words_total = 0
        self.__bow = {}
        
    # Adding two BoWs.
    def __add__(self, diff):
        a = bow()
        sum = a.__bow
        for key in self.__bow:
            sum[key] = self.__bow[key]
            if key in diff.__bow:
                sum[key] += diff.__bow[key]
        for key in diff.__bow:
            if key not in sum:
                sum[key] = diff.__bow[key]
        return a
        
    # Adding a word in the BoW. 
    def add_word(self, word):
        self.__words_total += 1
        if word in self.__bow:
            self.__bow[word] += 1
        else:
            self.__bow[word] = 1
    
	# Number of words in the bag, i.e. length of BoW.
    def len(self):
        return len(self.__bow)
    
    # List of the words contained in the bag.
    def word_list(self):
        return self.__bow.keys()
    
    # BoW with words and their frquencies.
    def bow(self):
        return self.__bow
        
    # Frequency of a word.
    def freq(self, word):
        if word in self.__bow:
            return self.__bow[word]
        else:
            return 0

# This class is to manipulate training data and testing data.
class dataset(object):
 
    # Initialization.
    def __init__(self, voc):
        self.__dataset_class = None
        self._bag = bow()
        dataset._voc = voc
    
	# Training data = true, testing data = false.
    def read_data(self, input_file, training = False):
        words = open(input_file, "r").read()
        words = words.lower()
        words = re.split(r"\W", words)
        self._words_total = 0
        for word in words:
            self._bag.add_word(word)
            if training:
                dataset._voc.add_word(word)

    # Joining two datasets.
    def __add__(self, diff):
        a = dataset(dataset._voc)
        a._bag = self._bag + diff._bag    
        return a
    
    def voc_len(self):
        return len(dataset._voc)
                
    # Words and their frequency with the BoW feature.
    def dictionary(self):
        return self._bag.bow()
        
    def word_list(self):
        d =  self._bag.bow()
        return d.keys()
    
    def freq(self, word):
        bow =  self._bag.bow()
        if word in bow:
            return bow[word]
        else:
            return 0
                
    # Intersection of two dotasets.
    def __and__(self, diff):
        inter = []
        w = self.word_list()
        for word in diff.word_list():
            if word in w:
                inter += [word]
        return inter

# Class for dataset labels.
class datasetClass(dataset):

    def __init__(self, voc):
        dataset.__init__(self, voc)
        self._dataset_size = 0

    # Calculate the probabilty of a word.
    def probability(self, word):
        voc_len = dataset._voc.len()
        sum = 0
        for i in range(voc_len):
            sum = datasetClass._voc.freq(word)
        prob = 1 + self._bag.freq(word)
        prob /= voc_len + sum
        return prob

    # Adding two datasetClass objects.
    def __add__(self, diff):
        a = datasetClass(self._voc)
        a._bag = self._bag + diff._bag 
        return a

    def noofDataset(self, number):
        self._dataset_size = number
    
    def datasetSize(self):
        return self._dataset_size

# Train the traning data and predic the testing data + Laplace smoothing (add-1).
class naiveB(object):

    pred_label_list = []
    pred_prob_list = []

    def __init__(self):
        self.__dataset_cl = {}
        self.__voc = bow()
            
    # The number of times unique words in a label class.
    def words_total(self, label_class):
        sum = 0
        for word in self.__voc.word_list():
            dict = self.__dataset_cl[label_class].dictionary()
            if word in dict:
                sum +=  dict[word]
        return sum
    
    def training(self, directory, dclass_name):
        v = datasetClass(self.__voc)
        dir = os.listdir(directory)
        for f in dir:
            d = dataset(self.__voc)
            d.read_data(directory + "/" +  f, training = True)
            v = v + d
        self.__dataset_cl[dclass_name] = v
        v.noofDataset(len(dir))

    # Predict the testing dataset with Laplace smoothing (add-1).
    def predict(self, data, label_class = ""):
        if label_class:
            sum_dclass = self.words_total(label_class)
            prob = 0
            d = dataset(self.__voc)
            d.read_data(data)

            for j in self.__dataset_cl:
                sum_ind = self.words_total(j)
                prod = 1
                for i in d.word_list():
                    wf_dclass = 1 + self.__dataset_cl[label_class].freq(i) # Laplace smoothing (add-1).
                    wf = 1 + self.__dataset_cl[j].freq(i)                  # Laplace smoothing (add-1).
                    r = wf * sum_dclass / (wf_dclass * sum_ind)            # Laplace smoothing (add-1).
                    prod *= r
                prob += prod * self.__dataset_cl[j].datasetSize() / self.__dataset_cl[label_class].datasetSize()
            if prob != 0:
                return 1 / prob
            else:
                return -1
        else:
            overall_list = []
            for label_class in self.__dataset_cl:
                prob = self.predict(data, label_class)
                overall_list.append([label_class, prob])
            for k in overall_list:
                if (k[1] > 0.5):
                    self.pred_label_list.extend([k[0]])
                    self.pred_prob_list.extend([k[1]])
            return overall_list
