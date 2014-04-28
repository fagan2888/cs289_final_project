import math
import random
import numpy as np
import util

class Node(object):

    def __init__(self, crash_indexes):
        #the crash indexes of this node and its subnodes
        self.mi = crash_indexes
        self.left = None
        self.right = None
        self.splitting_feature = None
        self.splitting_threshold = None
        self.most_common_label = None
        self.impurity = None

    def get_crashes(self):
        return [Node.crashes[i] for i in self.mi]

    def get_labels(self):
        return [Node.labels[i] for i in self.mi]

    def get_impurity(self):
        if self.impurity is None:
            self.impurity = calc_impurity(self.get_labels())
        return self.impurity

    #Return the most common label among the values in this node
    #and its subnodes
    def get_most_common_label(self):
        #Caching the most common label to improve performance
        if self.most_common_label is None:
            labels = self.get_labels()
            spam_count = labels.count(1)
            #hardcoded references to labels of 0 and 1. not ideal!
            if spam_count>len(labels)/2.0:
                self.most_common_label = 1
            elif spam_count == len(labels)/2.0:
                #If we can't tell, randomize to match the typical spam %
                if random.random()<=Node.spam_proportion:
                    self.most_common_label = 1
                else:
                    self.most_common_label = 0
            else:
                self.most_common_label = 0
        return self.most_common_label

    def predicted_label(self, crash):
        if self.splitting_feature is None:
            return self.get_most_common_label()
        if crash[self.splitting_feature]>=self.splitting_threshold:
            return self.left.predicted_label(crash)
        return self.right.predicted_label(crash)

    def split(self, feature_consideration_limit):
        crashes = self.get_crashes()
        labels = self.get_labels()
        #Limit the features we consider to generate more random trees
        split_thresholds = self.calc_split_thresholds(crashes, labels,
                feature_consideration_limit)
        splitting_feature, splitting_threshold, split_type = self._calc_best_split(
                split_thresholds, crashes, labels)
        #if the node could not be split 
        if splitting_feature == -1:
            return None, None, None, None, None
        left_node_indexes, right_node_indexes = self.split_on_feature(splitting_feature, splitting_threshold, crashes)
        return left_node_indexes, right_node_indexes, splitting_feature, splitting_threshold, split_type

    def _calc_best_split(self, split_thresholds, crashes, labels):
        current_impurity = self.get_impurity()
        best_impurity_drop = 0
        best_feature_index = -1
        best_split_value = 0
        best_split_type = -1
        #for each feature
        for i, split_thresholds_i in enumerate(split_thresholds):
            for j, split_value in enumerate(split_thresholds_i):
                left_spam, left_size, right_spam, right_size = self.split_on_feature_spam_only(i, split_value, crashes, labels)
                impurity_drop = calc_impurity_drop_spam_only(current_impurity, left_spam, left_size, right_spam, right_size)
                #print i, 'impurity drop', impurity_drop, left_spam
                if impurity_drop>best_impurity_drop:
                    best_impurity_drop = impurity_drop
                    best_feature_index = i
                    best_split_value = split_value
                    best_split_type = j
        return best_feature_index, best_split_value, best_split_type

    def calc_split_thresholds(self, crashes, labels,
            feature_consideration_limit, search_type='edge'):
        if feature_consideration_limit is None:
            features_to_consider = xrange(Node.feature_count)
        else:
            features_to_consider = util.randsample(Node.feature_count, 
                    feature_consideration_limit)
        split_thresholds = list()
        #f represents which feature we're evaluating
        #Normally, f ranges from 0 to 57
        for f in features_to_consider:
            feature_f = [crash[f] for crash in crashes]
            if search_type == 'edge':
                #The set of features from each crash with that label
                spam_f = list()
                nonspam_f = list()
                #i is which crash we're looking at
                for i, feature in enumerate(feature_f):
                    if labels[i]:
                        spam_f.append(feature)
                    else:
                        nonspam_f.append(feature)
                split_thresholds.append([mina(spam_f), maxa(spam_f),
                    mina(nonspam_f), maxa(nonspam_f), util.avg(feature_f)])

            elif search_type=='fixed':
                split_thresholds.append(np.linspace(min(feature_f), 
                    max(feature_f), 9))
                    
            elif search_type=='mean':
                average = util.avg(feature_f)
                #minimum never helps for some reason
                minimum = min(feature_f)
                maximum = max(feature_f)
                avg_low = util.avg([average, minimum])
                avg_high = util.avg([average, maximum])
                split_thresholds.append([average])
            elif search_type=='median':
                feature_f.sort()
                split_thresholds.append([get_median_of_sorted(feature_f)])
            else:
                print 'invalid average type', search_type
                return False
        return split_thresholds

    #To calculate the best split, we don't need the full complexity of the node
    #We just need the impurity of the node, which we can get with just the
    # proportion of spam crashes in the node, which can give us the impurity
    def split_on_feature_spam_only(self, feature_index, threshold, crashes,
            labels):
        left_spam, right_spam = 0, 0
        left_size, right_size = 0, 0
        for i, crash in enumerate(crashes):
            if crash[feature_index]>=threshold:
                left_spam+=labels[i]
                left_size+=1
            else:
                right_spam+=labels[i]
                right_size+=1
        if left_size:
            left_spam /= float(left_size)
        if right_size:
            right_spam /= float(right_size)
        return left_spam, left_size, right_spam, right_size

    def split_on_feature(self, feature_index, threshold, crashes):
        left_node_indexes, right_node_indexes = list(), list()
        #http://stackoverflow.com/questions/949098/python-split-a-list-based-on-a-condition
        for i, crash in enumerate(crashes):
            if crash[feature_index]>=threshold:
                left_node_indexes.append(self.mi[i])
            else:
                right_node_indexes.append(self.mi[i])
        return left_node_indexes, right_node_indexes

    def __repr__(self):
        return 'Node('+str(self.mi)+')'

#Take a list of labels for each index in the node
def calc_impurity(node_labels, impurity_type='entropy'):
    if len(node_labels)==0:
        return 0
    #proportion of nonspam crashes
    spam_proportion = node_labels.count(1)/float(len(node_labels))
    return calc_impurity_spam_only(spam_proportion)

#take as input the proportion of spam in the node
def calc_impurity_spam_only(spam, impurity_type='entropy'):
    notspam = 1 - spam
    if impurity_type=='entropy':
        #If there are any crashes which are not spam
        #Do the entropy calculation on them
        if spam:
            spam = spam*math.log(spam, 2)
        if notspam:
            notspam = notspam*math.log(notspam, 2)
        return -(spam+notspam)
    if impurity_type=='variance':
        return notspam*spam
    else:
        print 'Invalid impurity type', impurity_type
        return False
"""
def calc_impurity_drop(current_impurity, left_node, right_node, labels):
    left_impurity = calc_impurity(left_node, labels)
    right_impurity = calc_impurity(right_node, labels)
    #weight by proportion of patterns that end up in each node
    left_proportion = len(left_node)/float(len(left_node)+len(right_node))
    left_impurity *= left_proportion
    right_impurity *= 1-left_proportion
    #sum up the result
    return current_impurity - left_impurity - right_impurity
"""

def calc_impurity_drop_spam_only(current_impurity, left_spam, left_size, right_spam, right_size):
    left_impurity = calc_impurity_spam_only(left_spam)
    right_impurity = calc_impurity_spam_only(right_spam)
    #weight by proportion of patterns that end up in each node
    left_proportion = left_size/float(left_size+right_size)
    left_impurity *= left_proportion
    right_impurity *= 1-left_proportion
    return current_impurity - left_impurity - right_impurity

#Min approximate: subtract a small threshold value to ensure that 
#> comparisons will include the min
def mina(x, threshold=10e-10):
    return min(x)-threshold

def maxa(x, threshold=10e-10):
    return max(x)+threshold
