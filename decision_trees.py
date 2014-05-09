import math
import numpy as np
import decision_nodes
import util

def construct_decision_tree_iterative(node, crashes, labels,
        feature_consideration_limit):
    unsplit_nodes = [node]
    split_type_counts = [0 for i in xrange(5)]
    while len(unsplit_nodes)>0:
        current_node = unsplit_nodes[0]
        if current_node.get_impurity() == 0:
            del(unsplit_nodes[0])
            continue
        left_crash_indexes, right_crash_indexes, splitting_feature, splitting_threshold, split_type = current_node.split(feature_consideration_limit)
        #if the node could not be split
        if left_crash_indexes is None:
            del(unsplit_nodes[0])
            continue
        #split_type_counts[split_type]+=1
        left_node = decision_nodes.Node(left_crash_indexes)
        right_node = decision_nodes.Node(right_crash_indexes)
        current_node.left = left_node
        current_node.right = right_node
        current_node.splitting_feature = splitting_feature
        current_node.splitting_threshold = splitting_threshold
        unsplit_nodes.append(current_node.left)
        unsplit_nodes.append(current_node.right)
        del(unsplit_nodes[0])
    #print split_type_counts
    return node

def calc_error_rate(tree, crashes, labels):
    error_count = 0
    for i in xrange(len(crashes)):
        if not tree.predicted_label(crashes[i]) == labels[i]:
            error_count += 1
    return error_count/float(len(crashes))

def calc_error_rate_forest(forest, crashes, labels):
    error_count = 0
    predictions = list()
    for i, crash in enumerate(crashes):
        predicted_label = predict_label_forest(forest, crash)
        if not predicted_label == labels[i]:
            error_count += 1
        predictions.append(predicted_label)
    return error_count/float(len(crashes)), predictions

def calc_error_rate_forest_weighted(forest, crashes, labels, weights):
    error_count = 0
    predictions = list()
    for i, crash in enumerate(crashes):
        predicted_label = predict_label_forest(forest,crash)
        if not predicted_label == labels[i]:
            error_count += weights[i]
        predictions.append(predicted_label)
    return error_count, predictions
    

def assign_labels(tree, x_test, outputfile_name):
    outputfile = open(outputfile_name, 'w')
    outputfile.write('Id,Category\n')
    for i, crash in enumerate(x_test):
        outputfile.write('{0},{1}\n'.format(i+1, tree.predicted_label(crash)))

def assign_labels_forest(forest, x_test, outputfile_name):
    outputfile = open(outputfile_name, 'w')
    outputfile.write('Id,Category\n')
    for i, crash in enumerate(x_test):
        outputfile.write('{0},{1}\n'.format(i+1, predict_label_forest(forest,crash)))

def assign_labels_adaboost(forests, alpha, x_test, outputfile_name):
    outputfile = open(outputfile_name, 'w')
    outputfile.write('Id,Category\n')
    for i, crash in enumerate(x_test):
        outputfile.write('{0},{1}\n'.format(i+1, predict_label_adaboost(
            forests, alpha, crash)))

def construct_tree(crashes, labels, use_iterative, feature_consideration_limit):
    initial_node = decision_nodes.Node(range(len(crashes)))
    if use_iterative:
        return construct_decision_tree_iterative(initial_node, crashes, labels,
                feature_consideration_limit)
    #If not iterative, assume recursive
    return construct_decision_tree(initial_node, crashes, labels)

def construct_forest(crashes, labels, forest_size, tree_size, 
        weights = None, randomize_first=True, feature_consideration_limit = None):
    forest = list()
    for i in xrange(forest_size):
        if i==0 and not randomize_first:
            crash_indexes = range(tree_size)
        else:
            crash_indexes = util.randresample(len(crashes), tree_size, 
                    weights = weights)
        #initial_node = decision_nodes.Node(range(tree_size))
        initial_node = decision_nodes.Node(crash_indexes)
        forest.append(construct_decision_tree_iterative(initial_node,
            crashes, labels, feature_consideration_limit))
    return forest

def predict_label_forest(forest, crash):
    predicted_labels = list()
    for tree in forest:
        predicted_labels.append(tree.predicted_label(crash))
    #Interestingly, > gives better performance than >=
    if np.count_nonzero(predicted_labels) > len(predicted_labels)/2:
        return 1
    return 0

def adaboost_forest(crashes, labels, rounds, forest_size,
        tree_size, crashes_validate, labels_validate,
        iter_validate):
    initial_weight = 1/float(len(crashes))
    weights = [initial_weight for i in crashes]
    forests = list()
    alpha = list()
    for t in xrange(rounds):
        feature_consideration_limit = decision_nodes.Node.feature_count
        error = 1
        #weighted_crashes, weighted_labels = get_weighted_training_data(
        #        crashes, labels, weights)
        while error>=0.5:
            forest = construct_forest(crashes, labels, forest_size,
                    tree_size, weights, feature_consideration_limit)
            error_check_indexes = util.randresample(len(crashes),
                    weights = weights)
            error_check_crashes = [crashes[i] for i in error_check_indexes]
            error_check_labels = [labels[i] for i in error_check_indexes]
            error, unused_predictions = calc_error_rate_forest(forest,
                    error_check_crashes, error_check_labels)
            if error>=0.5:
                print 'Not a weak learner! fcl:', feature_consideration_limit, 't:', t, 'error:', error
            #If we're struggling to figure out the right step  
            #The existing feature splitting may be causing prolblems 
            #So use a more random analysis of features to find new options 
            feature_consideration_limit-=1
            if feature_consideration_limit == 0:
                feature_consideration_limit = decision_nodes.Node.feature_count

        forests.append(forest)
        #If error is zero, unclear what to do
        if error==0:
            alpha.append(max(alpha))
        else:
            alpha.append(math.log((1-error)/error)/2)
        unused_error, predictions = calc_error_rate_forest(forest, 
                crashes, labels)
        for i in xrange(len(weights)):
            if predictions[i] == labels[i]:
                weights[i] *= math.exp(-alpha[t])
            else:
                weights[i] *= math.exp(alpha[t])
        #print t, predictions[0], util.prettyf(weights[0])
        #normalize so the weights are a distribution
        #that is, so the sum of the weights is 1
        normalization_factor = 1/sum(weights)
        weights = [weight*normalization_factor for weight in weights]
        if iter_validate and (t+1)%iter_validate==0:
            error_rate, predictions = calc_error_rate_adaboost([[forests, alpha]],
                    crashes_validate, labels_validate)
            print '@t', t,'error', error_rate, 'alpha', alpha[t]
    return forests, alpha

def predict_label_adaboost(forests, alpha, crash):
    total = 0
    for t, forest in enumerate(forests):
        prediction = predict_label_forest(forest, crash)
        #convert prediction to {-1, 1} for math
        if prediction==0: prediction=-1
        total += alpha[t]*prediction
    if total>=0:
        return 1
    return 0

def calc_error_rate_adaboost(adaboosts, crashes, labels):
    error_total = 0
    predictions = list()
    for i, crash in enumerate(crashes):
        prediction_sum = 0
        #get the average prediction among all adaboosts 
        for adaboost in adaboosts:
            prediction_sum += 1 if predict_label_adaboost(adaboost[0], adaboost[1], crash) else -1
        #There should be an odd number of adaboosts to avoid ties!
        #Ties which do exist are assumed to be nonspam
        if prediction_sum>0:
            predictions.append(1)
        else:
            predictions.append(0)
        if not predictions[i] == labels[i]:
            error_total+=1
    return error_total/float(len(crashes)), predictions

def do_stuff(crashes, labels, crashes_validate, labels_validate, x_test, args):
    decision_nodes.Node.crashes = crashes
    decision_nodes.Node.labels = labels
    decision_nodes.Node.feature_count = len(crashes[0])
    decision_nodes.Node.spam_proportion = np.count_nonzero(labels)/float(len(labels))

    if args['method'] == 'tree':
        tree = construct_tree(crashes, labels, use_iterative = True,
                feature_consideration_limit = None)
        print 'tree error', calc_error_rate(tree, crashes_validate, 
                labels_validate)
        if args['output']:
            assign_labels(tree, x_test, args['output'])

    elif args['method'] == 'forest':
        forest = construct_forest(crashes, labels,
                forest_size = args['forest_size'], tree_size = args['tree_size'],
                randomize_first=False)
        error_rate, predictions = calc_error_rate_forest(forest,
                crashes_validate, labels_validate)
        print 'forest error', error_rate
        if args['output']:
            assign_labels_forest(forest, x_test, args['output'])

    elif args['method'] == 'adaboost':
        adaboosts = list()
        for i in xrange(args['ada_size']):
            #if i>0:
            #    crashes, labels = util.shuffle(crashes, labels)
            forests, alpha = adaboost_forest(crashes, labels, 
                    rounds = args['iterations'], forest_size = args['forest_size'], 
                    tree_size = args['tree_size'],
                    crashes_validate = crashes_validate,
                    labels_validate = labels_validate,
                    iter_validate = args['iter_validate'])
            adaboosts.append([forests, alpha])
            print 'adaboost', i, 'calculated'
            if args['output']:
                assign_labels_adaboost(forests, alpha, x_test,
                        args['output']+str(i))


        error_rate, predictions =  calc_error_rate_adaboost(adaboosts, 
                    crashes_validate, labels_validate)
        print '@t', args['iterations'], 'error rate', error_rate
        if args['output']:
            assign_labels_adaboost(forests, alpha, x_test,
                    args['output'])
