#http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
import scipy.io as spio
import numpy as np

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def import_mnist_training_data(filename, subset_count):
    full_data = loadmat(filename)
    images = [None]*subset_count
    labels = [None]*subset_count
    for i in xrange(subset_count):
        subset_data = full_data['train'][i].__dict__
        #We need to transpose the image data so we have 100 sets of 28x28 images
        images[i] = np.rollaxis(subset_data['images'], 2, 0)
        labels[i] = subset_data['labels']
    return images, labels
    
def import_mnist_testing_data(filename):
    full_data = loadmat(filename)
    images = np.rollaxis(full_data['test']['images'], 2, 0)
    labels = full_data['test']['labels']
    return images, labels

def import_spam_data(filename):
    full_data = loadmat(filename)
    x_train = full_data['Xtrain']
    y_train = full_data['ytrain']
    x_test = full_data['Xtest']
    return x_train, x_test, y_train
