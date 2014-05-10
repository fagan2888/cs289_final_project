import bisect
import random
import numpy as np
import cPickle

def count_unique_values(iterable):
    temp = set()
    result = list()
    for x in iterable:
        temp.add(x)
    return len(temp)

def get_median_of_sorted(input_list):
    median_index = len(input_list)/2
    return input_list[median_index]

#http://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
def randresample(option_count, limit = None, weights = None):
    if limit is None:
        limit = option_count
    if weights is None:
        #Easy way
        selections = list()
        for i in xrange(limit):
            selections.append(random.randint(0,option_count-1))
        return selections
    #Hard way, with weights
    #convert the weights into cumulative weights for math
    cumulative_weights = []
    total = 0
    for weight in weights:
        total += weight
        cumulative_weights.append(total)
    #total should be 1 if weights is a proper distribution
    selections = list()
    for i in xrange(limit):
        #get a random weight and find the associated index 
        rnd = random.random()*total
        j = bisect.bisect(cumulative_weights, rnd)
        selections.append(j)
    return selections

def randsample(option_count, limit = None):
    options = range(option_count)
    random.shuffle(options)
    if limit is None:
        return options
    return options[:limit]

#shuffle two lists in the same way
def shuffle(list1, list2, to_numpy_array = False):
    if not len(list1)==len(list2):
        print 'Error: lists not of equal size'
        print len(list1), len(list2)
        return False
    image_count=len(list1)
    combined = zip(list1,list2)
    random.shuffle(combined)
    list1_shuffled, list2_shuffled = zip(*combined)
    if to_numpy_array:
        list1_shuffled = np.array(list1_shuffled)
        list2_shuffled = np.array(list2_shuffled)
    return list1_shuffled, list2_shuffled

def printf(float_list):
    print [prettyf(x) for x in float_list]

def prettyf(num):
    return "%0.2e" % num

def import_cyclist_data(input_filename):
    input_file = open(input_filename, 'rb')
    #pandas dataframe
    df = cPickle.load(input_file)
    y = df['SER_INJ'].values
    df = df.drop('SER_INJ', 1)
    #print df.info()
    x = df.values
    return x, y

#Import the 2011 and 2012 data intelligently, stripping out columns from 2011
#data that don't exist in 2012 data
def smart_import_cyclist_data(input_filename_2011, input_filename_2012):
    input_file_2011 = open(input_filename_2011, 'rb')
    input_file_2012 = open(input_filename_2012, 'rb')
    #pandas dataframe
    data_2011 = cPickle.load(input_file_2011)
    data_2012 = cPickle.load(input_file_2012)
    print data_2011.shape, data_2012.shape
    print type(data_2012.columns)
    shared_columns = [col for col in data_2011.columns.values
            if col in data_2012.columns.values]
    print len(shared_columns)
    #data_2011 = data_2011[data_2012.columns.values]
    data_2011 = data_2011[shared_columns]
    data_2012 = data_2012[shared_columns]
    print data_2011.shape, data_2012.shape
    x_train, y_train = extract_xy_from_cyclist_data(data_2011)
    x_test, y_test = extract_xy_from_cyclist_data(data_2012)
    return x_train, y_train, x_test, y_test

    
def extract_xy_from_cyclist_data(cyclist_data):
    y = cyclist_data['SER_INJ'].values
    cyclist_data = cyclist_data.drop('SER_INJ', 1)
    #print cyclist_data.info()
    x = cyclist_data.values
    return x, y
