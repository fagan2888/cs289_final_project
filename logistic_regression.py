import matio
import numpy as np
import math
import matplotlib.pyplot as pyplot
import random
import util

def init_argument_parser():
    parser = argparse.ArgumentParser('Homework 4')
    parser.add_argument('--problem', dest='problem', type=str,
            default='problem_1', choices=['p1', 'p2-1', 'p2-2', 'p2-3', 'p2-4'])
    return vars(parser.parse_args())

#because it is much more convenient for math, x is normally
#stored in design matrix format
def calc_hessian(design_matrix, mu, lam):
    mumu = [mu_i*(1-mu_i) for mu_i in mu]
    s = np.diag(mumu)
    inner = np.dot(s, design_matrix)
    main_hessian = np.dot(np.transpose(design_matrix), inner)
    regularization = np.multiply(2*lam, np.identity(len(main_hessian)))
    main_hessian = np.add(main_hessian, regularization)
    return main_hessian

#http://stackoverflow.com/questions/903853/how-to-extract-column-from-a-multi-dimentional-array
def get_col(train, col_index):
    return train[:,col_index]

#http://stackoverflow.com/questions/4544292/how-do-i-standardize-a-matrix
def standardize_col(col):
    mean = sum(col)/float(len(col))
    std_dev = np.std(col)
    col = [(value-mean)/std_dev for value in col]
    return col

def log_transform_col(col):
    return [math.log(value + 0.1) for value in col]

def binarize_col(col):
    return [1 if value>0 else 0 for value in col]

#Augment each feature vector
def augment_feature_vectors(x):
    return np.transpose(np.insert(np.transpose(x), 0, 1, axis=0))

def process_columns(training_data, function, augment=True):
    data_count = len(training_data)
    col_count = len(training_data[0])
    #copy the data to avoid overwriting the original
    processed_data = np.empty(shape=training_data.shape)
    for i in xrange(col_count):
        col = get_col(training_data, i)
        processed_col = function(col)
        for j, processed_value in enumerate(processed_col):
            processed_data[j][i] = processed_value
    if augment:
        processed_data = augment_feature_vectors(processed_data)
    return processed_data

def standardize_data(training_data):
    return process_columns(training_data, standardize_col)

def log_transform_data(training_data):
    return process_columns(training_data, log_transform_col)

def binarize_data(training_data):
    return process_columns(training_data, binarize_col)

#negative log likelihood. use lam instead of lambda because
#lambda is a python keyword
#x_i \in R^d, y_i \in {0,1}
def calc_nll(x, y, beta, mu, lam):
    total = 0
    for i in xrange(len(x)):
        total-= y[i]*math.log(mu[i])
        total-= (1-y[i])*math.log(1-mu[i])
    #then add regularization parameter
    total += lam*np.dot(beta, beta)
    return total

def calc_mu(x, beta):
    x_dot_beta = x.dot(beta)
    mu = [1/(1+np.exp(-x_dot_beta[i])) for i in xrange(x_dot_beta.shape[0])]
    return mu

def calc_gradient(design_matrix, y, beta, mu, lam):
    sub = np.subtract(mu, y)
    main_gradient = np.dot(np.transpose(design_matrix), sub)

    main_gradient += 2*lam*beta
    return main_gradient

#approximate the gradient based on one data point
#x_i is d+1 dimensional vector, y_i and mu_i are scalars
def calc_single_gradient(x_i, y_i, beta, mu_i, lam):
    sub = mu_i-y_i
    main_gradient = sub*x_i

    regularization = [2*lam*beta_i for beta_i in beta]
    main_gradient+=regularization
    return main_gradient

def newton_iteration(beta, hessian_inverse, gradient):
    return np.subtract(beta, np.dot(hessian_inverse, gradient))

def calc_beta(beta, step_size, gradient, iteration=0, scale_step_size=False):
    if scale_step_size:
        #If you don't multiply by 10, the step size is too small to get close
        step_size = step_size*10/float(iteration)
    return np.subtract(beta,step_size*gradient)

#calculating nll is not always necessary, and is huge performance killer
def run_batch_gradient_descent(nll, beta, x, y, lam, step_size, iterations, weight_step, should_calc_nll=False):
    mu = calc_mu(x, beta)
    if should_calc_nll:
        nll[0] = calc_nll(x, y, beta, mu, lam)
    for i in xrange(1, iterations):
        gradient = calc_gradient(x, y, beta, mu, lam)
        beta = calc_beta(beta, step_size, gradient, i, weight_step)
        mu = calc_mu(x, beta)
        if should_calc_nll:
            nll[i] = calc_nll(x, y, beta, mu, lam)

"""
#nll_limit limits the frequency with which we calculate the NLL
#Useful because it crushes performance and we only need to understand
#NLL in broad strokes
def run_stochastic_gradient_descent(x, y, lam, step_size, iterations,
        weight_step, nll_limit=1):
    feature_count = len(x[0])
    data_count = len(x)
    beta = [0 for i in xrange(feature_count)]
    nll = list()
    for loop in xrange(iterations/data_count+1):
        x, y = util.shuffle(x, y)
        x_curr = x[0]
        y_curr = y[0]
        mu = calc_mu(x, beta)
        nll.append(calc_nll(x, y, beta, mu, lam))
        for i in xrange(1, iterations-loop*data_count):
            full_iteration_count = loop*data_count+i
            x_curr = x[i]
            y_curr = y[i]
            mu_curr = 1/(1+np.exp(-np.dot(beta, x_curr)))
            gradient = calc_single_gradient(x_curr, y_curr, beta, mu_curr, lam)
            if weight_step:
                beta = np.subtract(beta,
                        step_size*gradient*10.0/full_iteration_count)
            else:
                beta = np.subtract(beta, step_size*gradient)
            if i%nll_limit==0:
                mu = calc_mu(x, beta)
                nll.append(calc_nll(x, y, beta, mu, lam))
                #print i/nll_limit, 'nll', nll[len(nll)-1]
    return nll, beta
"""

def plot_nll_data(nll_data, label, show=True):
    pyplot.plot(nll_data, label=label)
    if show:
        pyplot.xlabel('Iterations')
        pyplot.ylabel('NLL')
        pyplot.legend()
        pyplot.show()

def plot_batch_gradient_descent(x_train, y_train, lam=10, step_size=0.00001, 
        iterations=100, weight_step=False):
    x_train_std = standardize_data(x_train)
    x_train_binary = binarize_data(x_train)
    x_train_log = log_transform_data(x_train)

    nll_log, beta_log = run_batch_gradient_descent(x_train_log, y_train,
            lam, step_size, iterations, weight_step, should_calc_nll=True)
    print 'nll_log', nll_log
    print 'beta_log', beta_log

    #lam=10, step_size = 0.001 works
    nll_std, beta_std = run_batch_gradient_descent(x_train_std, y_train,
            lam, step_size, iterations, weight_step, should_calc_nll=True)
    print 'nll_std', nll_std
    print 'beta_std', beta_std
    nll_binary, beta_binary = run_batch_gradient_descent(x_train_binary, y_train,
            lam, step_size, iterations, weight_step, should_calc_nll=True)
    print 'nll_binary', nll_binary
    print 'beta_binary', beta_binary

    plot_nll_data(nll_std, 'standardized')
    plot_nll_data(nll_binary, 'binarized')
    plot_nll_data(nll_log, 'log transformed')

def plot_stochastic_gradient_descent(x_train, y_train, lam=100,
        step_size=0.00001, iterations=1000, nll_limit=1, weight_step=False):
    x_train_binary = binarize_data(x_train)
    x_train_std = standardize_data(x_train)
    x_train_log = log_transform_data(x_train)

    nll_binary, beta_binary = run_stochastic_gradient_descent(
            x_train_binary, y_train, lam, step_size, iterations, 
            weight_step, nll_limit)
    plot_nll_data(nll_binary, label='binarized')

    nll_std, beta_std = run_stochastic_gradient_descent(
            x_train_std, y_train, lam, step_size, iterations, weight_step, 
            nll_limit)
    plot_nll_data(nll_std, label='standardized')

    nll_log, beta_log = run_stochastic_gradient_descent(
            x_train_binary, y_train, lam, step_size, iterations,
            weight_step, nll_limit)
    plot_nll_data(nll_log, label='log transformed')

    plot_nll_data(nll_binary, label='binarized', show=False)
    plot_nll_data(nll_std, label='standardized', show=False)
    plot_nll_data(nll_log, label='log transformed')

def extract_fold(folded_list, fold_index, fold_count):
    length = len(folded_list)
    fold_start = (fold_index*length)/fold_count
    fold_stop = ((fold_index+1)*length)/fold_count
    fold = folded_list[fold_start:fold_stop]
    without_fold = np.concatenate((folded_list[:fold_start],
            folded_list[fold_stop:]))
    return without_fold, fold
    
def find_approximate_C_exp(images, labels, k, exp_range, **kwargs):
    best_C_exp=None

#All x and y data here comes from a training set
#the _train and _test labels are used in the context of the cross-validation
def cross_validate(x_full, y_full, lam, step_size, iterations, weight_step,
        k):
    #shuffle x_full and y_full so we can crossvalidate
    feature_count = len(x_full[0])
    x_full, y_full = util.shuffle(x_full, y_full, to_numpy_array = True)
    beta_all = np.zeros(shape=(k, feature_count))
    error_rates = np.empty(shape=k)
    nll = np.empty(shape=(k, iterations))
    for i in xrange(k):
        x_train, x_test = extract_fold(x_full, i, k)
        y_train, y_test = extract_fold(y_full, i, k)
        #This alters beta_all and possibly nll
        run_batch_gradient_descent(nll[i], beta_all[i], x_train, y_train, lam,
                step_size, iterations, weight_step)
        test_labels_calc = calc_labels(x_test, beta_all[i])
        error_rates[i] = calc_error_rate(test_labels_calc, y_test)
        print 'cross-validation error rate', error_rates[i]
    beta_all = sum(beta_all)/float(len(beta_all))
    print 'avg error rate', sum(error_rates)/float(len(error_rates))
    return beta_all
        
def calc_labels(x, beta):
    return [1 if np.dot(beta, x_i)>=0 else 0 for x_i in x]

def assign_labels(x_train, y_train, lam, step_size,
        iterations, k, weight_step=True, x_test = None):
    x_train = log_transform_data(x_train)
    #x_train = standardize_data(x_train)

    beta = cross_validate(x_train, y_train, lam, step_size, iterations,
            weight_step, k)

    labels_calc_train = calc_labels(x_train, beta)
    print 'training error rate', calc_error_rate(labels_calc_train,
            y_train)

    if x_test is not None:
        #Now work on the actual test data
        outputfile = open('x_test.txt', 'w')
        outputfile.write('Id,Category\n')
        x_train = log_transform_data(x_train)
        #x_test = standardize_data(x_test)

        labels_calc_test = calc_labels(x_test, beta)
        for i, label in enumerate(labels_calc_test):
            outputfile.write('{0},{1}\n'.format(i+1, label))

def calc_error_rate(labels_calculated, labels_truth):
    comparison = [labels_calculated[i]==labels_truth[i] for i in xrange(len(labels_calculated))]
    error_count = comparison.count(False)
    error_rate = error_count/float(len(labels_calculated))
    return error_rate
