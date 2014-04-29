import sys, numpy as np, pandas as pd
from copy import deepcopy

my_data=[['slashdot','USA','yes',18,'None'],
         ['google','France','yes',23,'Premium'],
         ['digg','USA','yes',24,'Basic'],
         ['kiwitobes','France','yes',23,'Basic'],
         ['google','UK','no',21,'Premium'],
         ['(direct)','New Zealand','no',12,'None'],
         ['(direct)','UK','no',21,'Basic'],
         ['google','USA','no',24,'Premium'],
         ['slashdot','France','yes',19,'None'],
         ['digg','USA','no',18,'None'],
         ['google','UK','no',18,'None'],
         ['kiwitobes','UK','no',19,'None'],
         ['digg','New Zealand','yes',12,'Basic'],
         ['slashdot','UK','no',21,'None'],
         ['google','UK','yes',18,'Basic'],
         ['kiwitobes','France','yes',19,'Basic']]

class decisionNode:
    def __init__(self, col= -1, value = None, results = None, tb = None, fb = None):
        self.col = col #col is the column that the tree splits on at this node
        #value is the value in the column that the tree splits on at this node
        self.value = value 
        #Results = a dictionary of the number of observations in each class
        self.results = results
        self.tb = tb #a tree grown on data meeting this node's splitting condition
        self.fb = fb #a tree grown on data failing this node's splitting condition
        return None

        
def divideset(rows, column, value):
    """rows = the array of observations
    column = the column of the feature vector to split the data on
    value = the value within column to split the data on.
    ==========
    Returns a tuple where the first object consists of data meeting the splitting
    condition (true) at a given node while the second object consists of data failing
    (false) to meet the splitting condition at the given node.
    """
    split_function = None #Initialize a variable split function.
    if isinstance(value, int) or isinstance(value, float): #Check if value is a number
        #True = the observation's value >= to the splitting criteria. False otherwise
        split_function = lambda row: row[column] >= value
    else:
        #If value is a string, True is where the observation's value == the criteria
        split_function = lambda row:row[column] == value
    
    #Divide the rows into two sets and return them
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)

def dividePandas(df, column, value):
    """df = a Pandas dataframe containing the observations used to build the CART
    column = the column the df is being split on.
    value = the value within column that the df is being split on.
    ==========
    Returns a tuple where the first object is the data meeting the splitting criteria
    and the second object is the data no meeting the splitting criteria."""
    if isinstance(value, int) or isinstance(value, float): #Check if value is a #
        #Divide the rows into two sets and return them
        set1 = df[df[column] >= value] #Observations greater than or equal to value
        set2 = df[df[column] < value] #Observations less than value are in set2
    else:
        set1 = df[df[column] == value] #Observations equal to value are in set 1
        set2 = df[df[column] != value] #Observations not equal to value are in set2 
    return (set1, set2)
    
def uniqueCounts(rows):
    """rows = a list of lists where the inner lists represent a row of data.
    Returns a dictionary of the results within "rows." The keys will be the 
    unique values within the "rows" results column. The values will be how many
    times the class appears within the results column of df."""
    results = {} #Initialize a dictionary to store the results
    for row in rows: #Iterate over all rows of data
        #The result is the last column
        r = row[-1]
        if r not in results: results[r] = 0 #Start the count for each class at zero
        results[r] += 1 #Increment the count for this row's class by 1
    return results

def uniqueCountsPandas(df, resCol):
    """df = the dataframe of data.
    resCol = the column name to get the results from.
    =========
    Returns a dictionary of the results within the dataframe. The keys will be the 
    unique values within the dataframe's results column. The values will be how many
    times the class appears within the results column of df."""
    return df.groupby(resCol).size().to_dict()
    
def giniImpurity(rows, resCol=None):
    """rows = the container of data for tree building. Either an list of lists or a
    pandas dataframe.
    resCol = None to use pure python to calculate gini Impurity OR
    restCol = a string denoting the column of results that you want to use with the 
    uniqueCountsPandas function to calculate gini-Impurity
    ==========
    Returns the giniImpurity of a given dataset."""
    if not resCol: #create the dictionary of counts for each class using pure python
        total = len(rows)
        counts = uniqueCounts(rows)
    else: #Create the dictionary of counts for each class using pandas.
        assert 'index' in dir(rows)
        total = len(rows)
        counts = uniqueCountsPandas(rows, resCol)
    imp = 1 #Initialize the gini-impurity at 1
    #Implement the formula for calculating gini-impurity
    fracs = [float(x)/total for x in counts.values()]
    for x in fracs:
        imp -= x*x
    return imp

def entropy(rows, resCol = None):
    """rows = the container of data for tree building. Either an list of lists or a
    pandas dataframe.
    resCol = None to use pure python to calculate gini Impurity OR
    restCol = a string denoting the column of results that you want to use with the 
    uniqueCountsPandas function to calculate gini-Impurity
    ==========
    Returns the entropy of a given dataset."""
    from math import log
    if not resCol: #create the dictionary of counts for each class using pure python
        total = len(rows)
        counts = uniqueCounts(rows)
    else: #Create the dictionary of counts for each class using pandas.
        assert 'index' in dir(rows)
        total = len(rows.index)
        counts = uniqueCountsPandas(rows, resCol)
    log2 = lambda x:log(x)/log(2) #Create a function to take log-base 2 of a number
    ent = 0 #Initialize the entropy at zero
    #Implement the formula for entropy, using log-base2
    fracs = [float(x)/total for x in counts.values()]
    for x in fracs:
        ent += -x*log2(x)
    return ent

def buildTree(rows, maxDepth = None, scoref=entropy, depth = 0):
    """rows = a list of lists where the inner lists represent a row of data.
    maxDepth = the maximum number of splits between the root node and a leaf node.
    scoref = the scoring function used to judge the performance of a given split.
    Currently can either be entropy or giniImpurity.
    depth = current depth of the tree. Used internally, do not alter.
    ==========
    Returns a decision tree of the specified maximum depth."""
    #A base condition for the recursion. Check if this branch of a split has no data
    if len(rows)==0:
        return decisionNode( )
    newDepth = depth + 1 #Calculate the depth of the next split.
    #Check if the depth at the next split is greater than a maximum specified depth
    if (maxDepth == 0 or maxDepth) and (newDepth > maxDepth): 
        return decisionNode(results=uniqueCounts(rows)) #If so, stop splitting.
    current_score=scoref(rows) #Calculate the current value of the score function.
    # Set up some variables to track the best criteria
    best_gain=0.0 #Initialize a value for the best gain from all possible splits
    best_criteria=None #Initialize a variable for the best column to split on
    best_sets=None #Initialize a variable for the best split's true and false data.

    #Count the number of columns in the row, minus the results column    
    column_count=len(rows[0])-1
    for col in range(0,column_count): #Iterate over all the columns of the data
        #Generate the list of different values in this column
        column_values={} #Initialize a dictionary to store the column values
        for row in rows: 
            #Iterate over each row, adding a key in the dict for each observed value
            column_values[row[col]]=1
        # Divide the dataset on each value in this column.
        for value in column_values.keys( ):
            (set1,set2)=divideset(rows,col,value)
            #Calculate the fraction of data in the true branch
            p=float(len(set1))/len(rows) 
            #Calculate the gain on the chosen score function using this split.
            gain=current_score-p*scoref(set1)-(1-p)*scoref(set2) 
            #Check if this split provides a better gain than the best previous split
            if gain>best_gain and len(set1)>0 and len(set2)>0:
                best_gain=gain
                best_criteria=(col,value)
                best_sets=(set1,set2)
    # Recursively create the subbranches
    if best_gain>0:
        trueBranch=buildTree(best_sets[0], maxDepth = maxDepth, depth = newDepth)
        falseBranch=buildTree(best_sets[1], maxDepth = maxDepth,  depth = newDepth)
        return decisionNode(col=best_criteria[0],value=best_criteria[1],
        tb=trueBranch,fb=falseBranch)
    else:
        return decisionNode(results=uniqueCounts(rows))

def printtree(tree,indent=''):
    # Is this a leaf node?
    if tree.results!=None:
        print str(tree.results)
    else:
        # Print the criteria
        print str(tree.col)+':'+str(tree.value)+'? '
        # Print the branches
        print indent+'T->',
        printtree(tree.tb,indent+' ')
        print indent+'F->',
        printtree(tree.fb,indent+' ')
        
def classify(observation,tree):
    """Returns the counts of the classes at the leaf"""
    if tree.results!=None:
        return tree.results
    else:
        v=observation[tree.col]
        branch=None
        if isinstance(v, int) or isinstance(v, float):
            if v>=tree.value:
                branch=tree.tb
            else: 
                branch=tree.fb
        else:
            if v==tree.value: 
                branch=tree.tb
            
            else: 
                branch=tree.fb
        return classify(observation,branch)

def classProbs(observation, tree, classes):
    """observation = a particular row from the design matrix
    tree = the particular tree that is being used to determine the probabilities
    classes = a list of possible outcomes/classes that we care about
    returns a list, in the same order as classes, that has the probabilities 
    of an observation being in each class"""
    res = classify(observation, tree) #res = results
    total = sum(res.values())
    probs = []
    for c in classes:
        if c in res.keys():
            probs.append(float(res[c])/total)
        else:
            probs.append(0)
    return probs

def predictWithTree(observation, tree, classes):
    """observation = a particular row from the dataframe
    tree = the tree used to give predictions
    classes = a list of possible outcomes/classes that we care about"""
    
    probs= classProbs(observation,tree, classes)
    return classes[np.argmax(probs)] 
    
def buildTreePandas(rows, res, maxDepth=None, scoref=entropy, depth=0):
    """rows = a list of lists where the inner lists represent a row of data.
    res = the column name of results
    maxDepth = the maximum number of splits between the root node and a leaf node.
    scoref = the scoring function used to judge the performance of a given split.
    Currently can either be entropy or giniImpurity.
    depth = current depth of the tree. Used internally, do not alter.
    ==========
    Returns a decision tree of the specified maximum depth."""
    if len(rows)==0: 
        return decisionNode( )
    newDepth = depth + 1
    if (maxDepth == 0 or maxDepth) and (newDepth > maxDepth):
        #print "Hooray I got here."
        return decisionNode(results=uniqueCountsPandas(rows, res))
    current_score=scoref(rows, resCol = res)
    # Set up some variables to track the best criteria
    best_gain=0.0
    best_criteria=None
    best_sets=None
    
    featColumns=rows.columns.tolist()
    featColumns.remove(res)
    for col in featColumns:
    # Generate the list of different values in
    # this column
        column_values=rows.loc[:,col].unique()
        # Now try dividing the rows up for each value
        # in this column
        copy = rows.sort(columns = col)
        for value in column_values:
            (set1,set2)=dividePandas(copy,col,value)
            # Information gain
            p=float(len(set1))/len(rows)
            gain=current_score-p*scoref(set1, resCol = res)-(1-p)*scoref(set2, resCol = res)
            if gain>best_gain and len(set1)>0 and len(set2)>0:
                best_gain=gain
                best_criteria=(col,value)
                best_sets=(set1,set2)
    # Create the subbranches
    if best_gain>0:
        trueBranch=buildTreePandas(best_sets[0], res, maxDepth = maxDepth, depth=newDepth)
        falseBranch=buildTreePandas(best_sets[1], res, maxDepth = maxDepth, depth=newDepth)
        return decisionNode(col=best_criteria[0],value=best_criteria[1],
        tb=trueBranch,fb=falseBranch)
    else:
        return decisionNode(results=uniqueCountsPandas(rows, res))
               
def searchFeats(tree):
    """tree= a completely built tree.
    ==========
    Returns a set of all the features actually used in the tree's branching.
    This function is used on an unpruned tree to identify relevant features."""
    if tree.results is None:
        #If I am at the branch, return my branching column and the columns of my
        #descendant branches.
        return set([tree.col]).union(searchFeats(tree.tb)).union(searchFeats(tree.fb))
    else: #If I am at a leaf, return an empty set
        return set()
      
def testAccuracy(testSet, tree, classes, res):
    preds = [] #Initialize an empty list to hold the predictions
    actual = np.array(testSet[res]) #Get the actual array of class labels
    for ind, row in testSet.iterrows(): #Iterate over all rows of the test data
        preds.append(predictWithTree(row, tree, classes)) #Make predictions on the test data
    preds = np.array(preds) #make predictions an array
    numWrong = 0 #Initialize the number of wrong predictions
    for i in range(len(preds)): #Iterate over all of the predictions and actual labels
        if preds[i] != actual[i]: #Check to see if the predictions are not the same.
            numWrong += 1 #If the predictions are not the same, increment numWrong by one
    return 1 - float(numWrong)/len(testSet) #Return the accuracy of the tree's predictions.
    
def get_results(node):
    """node = a particular decisionNode from a decision tree.
    ==========
    Returns a dictionary which has labels as keys, and the number of training observations with said label that made it to this 
    node."""
    if node.results is not None: #Check if the node is a leaf.
        return node.results #If the node is a leaf, then just return its results
    else: #If the node is not a leaf, recursively combine the results from its branches.
        tbr = get_results(node.tb) #get the results from the true branch
        fbr = get_results(node.fb) #get the results from the false branch
    
    new_results = deepcopy(tbr) #make a deep copy of the results from the true branch
    for key in fbr: #Iterate through the keys in the false branch
        if key not in new_results: #Check whether or not the key is in the deep copy of the true branch
            new_results[key] = fbr[key] #If it is not, add the key to the deep copy, and make its value equal to the value in fbr
        else: #If the key is in the deep copy of the true branch, add the false branch's count to the value in the deep copy
            new_results[key] = fbr[key] + new_results[key]
    return new_results #return the merged results from the false and true branches

def count_errors(node, testSet, res):
    """node = a particular decisionNode from within a given tree.
    testSet = the pandas dataframe of testing observations that would make it to this node
    res = the name of the column of results within the testSet dataframe.
    ==========
    Treats the node as a leaf and returns the number of testSet observations that would be misclassified on the basis of
    having made it to this node."""
    training_results = get_results(node) #Get a dictionary of labels and counts for the *training* data which made it to this node
    leaf_label = None #Initialize a label for this leaf
    majority_count = 0 #Initialize a variable to track the number of observations for the label with the most observations
    #Note that the steps below do not handle ties of the majority count in a nice way.
    for label, count in training_results.items(): #iterate through each pair of labels and counts from the training set
        if count > majority_count: #find the label with the highest count
            leaf_label = label #the label for the leaf is the label with the highest count
            majority_count = count #keep track of the count for the leaf_label
    
    wrong_labels = testSet[res].unique().tolist() #initialize wrong_labels to be all labels in the testSet
    if leaf_label in wrong_labels: #If the leaf label is in the list of labels for the part of the test set that got to this node
        wrong_labels.remove(leaf_label) #remove the leaf_label so that all which remains are incorrect labels
    
    wrong_count = 0 #Initialize a count of how many testSet observations will be classified incorrectly
    testCounts = testSet.groupby(res).size() #Get a series of the testSet labels and how many observations pertain to each label
    for label in wrong_labels: #Iterate over all the labels not equal to the leaf_label
        wrong_count += testCounts[label] #Sum up all of the observations with a label not equal to the leaf_label
    return wrong_count
    
def deep_count_errors(node, testSet, res):
    """node = a particular decisionNode from within a given tree.
    testSet = the pandas dataframe of testing observations that would make it to this node
    res = the name of the column of results within the testSet dataframe.
    ==========
    Distinguishes between branch nodes and leaf nodes. For leaf nodes, it returns the number of testSet observations that would be 
    misclassified on the basis of having made it to this node. For branch nodes, it returns the total number of observations that
    would be misclassified after making it to this node and being further classified by its descendant leaf nodes."""
    if node.results is not None: #Check if this node is a leaf node
        return count_errors(node, testSet, res) #If so, return the test set classification errors made by this node.
    else:
        tbSet = testSet[testSet[node.col] >= node.value] #find which test observations belong to this tree's true branch
        fbSet = testSet[testSet[node.col] < node.value] #find which test observations belong to this tree's false branch
        
        if node.tb.results is None: #Check if the true branch is a branch node
            #If so, get the count of all misclassifications made by this branch's descendent leaf nodes on the test observations
            term1 = deep_count_errors(node.tb, tbSet, res)
        else: #If the true branch is a leaf node, return the count of all test set classification errors made by the leaf.
            term1 = count_errors(node.tb, tbSet,res)
        if node.fb.results is None: #Check if the false branch is a branch node
            #If so, get the count of all misclassifications made by this branch's descendent leaf nodes on the test observations
            term2 = deep_count_errors(node.fb, fbSet, res)
        else: #If the false branch is a leaf node, return the count of all test set classification errors made by the leaf.
            term2 = count_errors(node.fb, fbSet, res) 
        return term1 + term2 #Sum the classification errors made by this nodes descendant leaves.

def prune(tree, testSet, res, technique):
    """tree = a decision tree to be pruned
    testSet = a pandas dataframe of observations and labels that were not used to train the tree
    res = the column name of the results column
    technique = a string indicating what pruning technique to use. Options include: "reduced_error".
    ==========
    Returns a decision tree that has been pruned according to a given pruning technique."""
    assert technique in ["reduced_error"]
    if technique == "reduced_error":
        tbSet = testSet[testSet[tree.col] >= tree.value] #find which test observations belong to this tree's true branch
        fbSet = testSet[testSet[tree.col] < tree.value] #find which test observations belong to this tree's false branch
        
        if tree.tb.results is None: #Check if the true branch of this sub-tree is a leaf
            ptb = prune(tree.tb, tbSet, res, technique) #If not, recursively travel down the true branch and prune it.
        else:
            ptb = tree.tb #If the true branch is a leaf, then the true branch has--in essence--already been pruned.
        if tree.fb.results is None: #Check if the false branch of this sub-tree is a leaf
            pfb = prune(tree.fb, fbSet, res, technique) #If not, recursively travel down the false branch and prune it.
        else:
            pfb = tree.fb #If the false branch is a leaf, then the false branch has--in essence--already been pruned.
        
        #Sum the number of misclassifications of the test data at each of the leaves of this node
        wrong_in_leaves = deep_count_errors(ptb, tbSet, res) + deep_count_errors(pfb, fbSet, res)
            
        #Count the number of misclassificationsof the test data that would occur if this node were treated as a leaf
        wrong_at_node = count_errors(tree, testSet, res)
        
        #Assess whether or not treating the node as a leaf improves the accuracy on the test set
        if wrong_at_node <= wrong_in_leaves: 
            #NOTE:The following line of code seems slightly redundant since count_errors(tree, testSet, res) had to call 
            #get_results(tree). I should set up some way to save the output of that function call instead of calling it twice.
            return decisionNode(results = get_results(tree)) #If so, return a decisionNode where the node is a leaf
        else:
            #If not, return a decisionNode where the node splits on the same column and value as before, but the 
            #true and false branches are the pruned-versions of the original true and false branches. See above for
            #definition of ptb and pfb
            return decisionNode(col = tree.col, value = tree.value, tb = ptb, fb = pfb)                  

def countNodes(tree):
    if tree.results is None:
        return countNodes(tree.tb) + countNodes(tree.fb)
    else:
        return 1

def forestPandas(data, resCol, maxDepth=None, percentage=70, numfeats = 15, fsize=5, selected=None):
    """data = a pandas dataframe of the feature vectors and the class.
    resCol = the string or number that names the column of results in 'data'.
    maxDepth = the maximum desired depth of the tree
    percentage = a number between 1 and 100 that represents the percentage of the data that you want to use for growing the trees
    numfeats = the number of features that you would like to randomly samply from the dataset.
    fsize = the size of the forest that you would like to great.
    ======================
    Returns a dictionary that contains the number of the tree along with the actual tree and the tree's data.
    """
    indices = data.index.tolist()
    trainingSets = {}
    percent = float(percentage)/100
    split = int(percent * len(indices) + 0.5)
    cols = data.columns.tolist()       
    for i in range(fsize + 1):
        if selected == None:
            np.random.shuffle(cols)
            selected = cols[:15]
            selected.append("spam")
        np.random.shuffle(indices)
        trainingSets[i] = {}
        trainingSets[i]["data"]= data[selected].loc[indices[:split + 1]]
        trainingSets[i]["tree"]= buildTreePandas(trainingSets[i]["data"], resCol, maxDepth=maxDepth) 
    return trainingSets

def ensembleVote(x, classes, ensemble):
    """x = a feature vector.
    classes = a list of classes that you want the vote to be between. 
    classes should be equal to or a subset of all unique values in the column of results.
    ensemble = the dictionary of trees, i.e. the forest, returned by treepredict.forestPandas"""
    votes = np.array([0 for kk in range(len(classes))])
    for i in ensemble:
        votes = votes + classProbs(x, ensemble[i]["tree"], classes)
    maxVote = 0
    loc = None
    for ind, vote in enumerate(votes):
        if vote > maxVote:
            maxVote = vote
            loc = ind
    prediction = classes[loc]
    return prediction