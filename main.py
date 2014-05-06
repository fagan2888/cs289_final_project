import argparse
from datetime import datetime
import cProfile
import util
import matio
import cPickle
import logistic_regression, decision_trees

def init_argument_parser():
    parser = argparse.ArgumentParser('CS289 Final Project')
    parser.add_argument('--method', dest='method', type=str,
            default='forest', choices=['tree', 'forest', 'adaboost', 'logistic',
                'logistic-plot'])
    parser.add_argument('-f', '--forest-size', dest='forest_size', type=int,
            default=1, help='How many trees are in each forest')
    parser.add_argument('-t', '--tree-size', dest='tree_size', type=int, default=0,
            help='The number of elements in each tree (NOT tree depth)')
    parser.add_argument('-a', '--ada-size', dest='ada_size', type=int, default=1,
            help='The number of adaboost implementations to run')
    parser.add_argument('--shuffle', dest='shuffle', type=int,
            default=1,
            help='Randomize the order of the crashes and labels?')
    parser.add_argument('-v', '--validate', dest='validate', type=int, 
            default = 500, help = 'Split the data into a training set and a validation set of the given size')
    parser.add_argument('--iter-validate', dest='iter_validate', type=int,
            default=50, help='Iteratively validate AdaBoost results after the given number of rounds')
    parser.add_argument('--input', dest='input', type=str,
            default='dataframes/design_DF_4Tree.pkl')
    parser.add_argument('--output', dest='output', type=str,
            default=None, help='Filename to output test results')
    parser.add_argument('-l', '--lambda', dest='lambda', type=float,
            default=100, help='lambda used for logistic regression')
    parser.add_argument('-s', '--step-size', dest='step_size', type=float,
            default=0.000005, help='step size used for logistic regression')
    parser.add_argument('-i', '--iterations', dest='iterations', type=int,
            default=100, help='iterations to run the chosen algorithm')
    parser.add_argument('--profile', dest='profile', action='store_true',
            default=False, help='Profile running time')
    parser.add_argument('-k', '--k-folds', dest='k', type=int, default=10,
            help='Number of k folds to use for cross validation')
    parser.add_argument('--beta-file', dest='beta_file', type=str, default=None,
            help='Import a precalculated beta from a file')
    return vars(parser.parse_args())

def run_logistic_regression(args):
    x_train, y_train = util.import_cyclist_data(args['input'])
    print 'cyclist data loaded'
    #x_train, x_test, y_train = matio.import_spam_data('spam.mat')
    if args['profile']:
        cProfile.runctx("logistic_regression.assign_labels(x_train, y_train, lam=args['lambda'], step_size = args['step_size'], iterations=args['iterations'], weight_step = False, k=args['k'], x_test = x_train)",
                {'logistic_regression': logistic_regression}, locals())
    else:
        if args['method'] == 'logistic-plot':
            logistic_regression.plot_batch_gradient_descent(x_train, y_train,
                    lam=args['lambda'], step_size = args['step_size'],
                    iterations = args['iterations'], weight_step = False)
        elif args['method'] == 'logistic':
            x_train = logistic_regression.log_transform_data(x_train)
            if args['beta_file'] is None:
                beta = logistic_regression.calc_cross_validated_beta(x_train,
                        y_train, lam=args['lambda'], 
                        step_size = args['step_size'],
                        iterations=args['iterations'], weight_step = False,
                        k=args['k'])
            else:
                inputfile = open(args['beta_file'], 'rb')
                beta = cPickle.load(inputfile)

            #Save beta
            beta_dumpfile = open('beta{0}{1}.pkl'.format(datetime.now().hour,
                datetime.now().minute), 'wb')
            cPickle.dump(beta, beta_dumpfile)

            labels = logistic_regression.calc_labels(x_train, beta)
            training_error = logistic_regression.calc_error_rate(labels, y_train)
            print 'training error rate', training_error
            logistic_regression.write_labels(x_train, beta)

def run_decision_trees(args):
    x_train, y_train = util.import_cyclist_data(args['input'])
    #x_train, x_test, y_train = matio.import_spam_data('spam.mat')

    if args['shuffle']:
        x_train, y_train = util.shuffle(x_train, y_train, to_numpy_array = True)

    if args['validate']>0:
        validation_size = args['validate']
        crashes = x_train[validation_size:]
        labels = y_train[validation_size:]
        crashes_validate = x_train[:validation_size]
        labels_validate = y_train[:validation_size]
    elif args['validate']==0:
        crashes = x_train
        labels = y_train
        crashes_validate = crashes
        labels_validate = labels
    else:
        #Check the training set error rate - useful for debugging 
        validation_size = args['validate']
        crashes = x_train[-validation_size:]
        labels = y_train[-validation_size:]
        crashes_validate = crashes
        labels_validate = labels

    x_test = crashes_validate

    if args['tree_size']==0:
        args['tree_size'] = len(crashes)

    if args['profile']:
        cProfile.runctx('decision_trees.do_stuff(crashes, labels, crashes_validate, labels_validate, x_test, args)', 
                {'decision_trees': decision_trees}, locals())
    else:
        decision_trees.do_stuff(crashes, labels, crashes_validate,
                labels_validate, x_test, args)

if __name__=="__main__":
    args = init_argument_parser()
    if args['method'].startswith('logistic'):
        run_logistic_regression(args)
    else:
        run_decision_trees(args)
