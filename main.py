import argparse
import cPickle
import cProfile
import util
import matio
import logistic_regression, decision_trees

def init_argument_parser():
    parser = argparse.ArgumentParser('CS289 Final Project')
    parser.add_argument('--method', dest='method', type=str,
            default='forest', choices=['tree', 'forest', 'adaboost', 'logistic'])
    parser.add_argument('-f', '--forest-size', dest='forest_size', type=int,
            default=1, help='How many trees are in each forest')
    parser.add_argument('-t', '--tree-size', dest='tree_size', type=int, default=0,
            help='The number of elements in each tree (NOT tree depth)')
    parser.add_argument('-r', '--rounds', dest='rounds', type=int, default=500,
            help='How many rounds to run the adaboost algorithm')
    parser.add_argument('-a', '--ada-size', dest='ada_size', type=int, default=1,
            help='The number of adaboost implementations to run')
    parser.add_argument('-s', '--shuffle', dest='shuffle', type=int,
            default=1,
            help='Randomize the order of the crashes and labels?')
    parser.add_argument('-v', '--validate', dest='validate', type=int, 
            default = 500, help = 'Split the data into a training set and a validation set of the given size')
    parser.add_argument('--iter-validate', dest='iter_validate', type=int,
            default=50, help='Iteratively validate AdaBoost results after the given number of rounds')
    parser.add_argument('-i', '--input', dest='input', type=str,
            default='dataframes/design_DF_4Tree.pkl')
    parser.add_argument('-o', '--output', dest='output', type=str,
            default=None, help='Filename to output test results')
    parser.add_argument('--profile', dest='profile', action='store_true',
            default=False, help='Profile running time')
    return vars(parser.parse_args())

def run_logistic_regression(args):
    x_train, x_test, y_train = matio.import_spam_data('spam.mat')
    if args['profile']:
        cProfile.runctx('logistic_regression.assign_labels(x_train, y_train, x_test, lam=100, step_size = 0.0005, iterations=100, k=10)',
                {'logistic_regression': logistic_regression}, locals())
    else:
        logistic_regression.assign_labels(x_train, y_train, x_test, lam=100,
                step_size = 0.0005, iterations=100, k=10)

def run_decision_trees(args):
    x_train, x_test, y_train = matio.import_spam_data('spam.mat')

    if args['shuffle']:
        x_train, y_train = util.shuffle(x_train, y_train, to_numpy_array = True)
    else:
        y_train = list(y_train)

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
    if args['method'] == 'logistic':
        run_logistic_regression(args)
    else:
        run_decision_trees(args)
