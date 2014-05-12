import numpy as np, pandas as pd, statsmodels.formula.api as smf
import treepredict, patsy, sys
from copy import deepcopy

def find_perfect_preds(nodes):
    """nodes = a list of nodes returned by calling treepredict.fetchnodes() on a
    particular decision tree.
    ==========
    Returns a list of the inices of nodes within the original list that was passed
    into the function, which perfectly predict the data i.e have only one class
    in their results."""
    
    #Initialize a list to store the indices of perfectly predicting nodes
    perfect_preds = []
    for place, node in enumerate(nodes): #Iterate over all the nodes in the list
        #Check if their is only one class in this node's results dictionary
        if len(node[-1].keys()) == 1: 
            #If so, place this node's number/index value in perfect_preds
            perfect_preds.append(place) 
    #Return the list of locations in 'nodes' with perfect predictors
    return perfect_preds
    
def reduce_nodeAugmented_4hybrid(orig_nodeAugmented_training, tree):
    """orig_nodeAugmented_training = a pandas dataframe with a "nodeNum" column
    tree = The tree providing the node numbers in orig_nodeAugmented_training
    ==========
    Returns a dataframe without the observations which belong to nodes with only
    one class in their results dictionary."""
    
    nodes = treepredict.fetchNodes(tree) #Get the node list for this tree
    #Get the list of perfectly predicting nodes within this tree
    perf_pred_nodes = find_perfect_preds(nodes)
    #Assign a variable to orig_nodeAugmented_training's column of node numbers 
    nodeNum = orig_nodeAugmented_training["nodeNum"]
    #Filter orig_nodeAugmented_training so that only those records which do NOT
    #belong to a perfectly predicting node 
    return orig_nodeAugmented_training[~nodeNum.isin(perf_pred_nodes)].copy()
    
def contrast_nodes_in_df(df):
    """df = a pandas dataframe that contains a single column of node numbers
    ==========
    Returns a pandas dataframe with an augmented series of contrast coded node
    numbers. There will be 1 column per node, and the original "nodeNum" column
    will also still be in the dataframe."""
    
    #Contrast the node column in design matrix for the logistic regression
    #Exclude the intercept
    contrast_nodeNums = patsy.dmatrix("C(nodeNum) - 1", data=df)
    #Extract the names that patsy provides for each column
    col_names = contrast_nodeNums.design_info.column_names
    #Convert contrast_nodeNums from a patsy dmatrix into a pandas dataframe
    #having the same index as df. This allows for merging
    contrast_nodeNums = pd.DataFrame(contrast_nodeNums, columns = col_names,
                                     index = df.index.tolist())
    #Merge the original df and contrast_nodeNums dataframe
    return pd.merge(df, contrast_nodeNums, left_index=True, right_index=True)
    
def incorporate_continuous_vars(df, missing_val):
    """df = a pandas dataframe where continuous variables containing observations
    with missing values are present
    missing_val = the value used in df to signify that the record has no data for
    the given variable
    ==========
    Returns a pandas dataframe with two additional columns per each continuous
    variable with missing data. 
    
    One column is a dummy variable for whether the
    record is missing data or not. The second column is the multiplication of the
    original column with the dummy variable.
    """
    #List the continuous variables with missing data. This is based on the specifics
    #of the NASS GES database being used
    continuous_vars = ["TRAV_SP", "NUMOCCS", "VSPD_LIM", "DRIVER_AGE"]
    
    #Iterate through all of the continuous_vars
    for variable in continuous_vars:
        #Initialize dummy variable columns for whether records are missing data
        df["{}_{}".format(variable, missing_val)] = 0
        for ind, row in df.iterrows(): #Iterate through the rows of df
            #Check if the row is actually missing data for the given variable
            if row[variable] == missing_val:
                #If so, set the value of the dummy variable column to 1
                df.loc[ind, variable] = 1
        #Create the interaction variable for the continous variable in question
        df["{}_interact_{}_{}".format(variable, variable, missing_val)] = df[variable] * df["{}_{}".format(variable, missing_val)]
    
    #Based on the arbitrary judgement that the number of lanes is more likely to
    #exhibit categorical like effects, this variable is contrast coded with the
    #reference value set to the missing value group
    
    #Create a design matrix with intercept of contrast coded VNUM_LAN's
    lane_dmatrix = patsy.dmatrix("C(VNUM_LAN, Treatment({}))".format(missing_val), data = df)
    
    #Isolate the column names from the patsy design matrix of VNUM_LAN's
    contrast_code_names = lane_dmatrix.design_info.column_names
    
    #Convert lane_dmatrix from a patsy dmatrix into a pandas dataframe
    #Use the index from df to facilitate merging.
    lane_dmatrix = pd.DataFrame(lane_dmatrix,
                                columns = contrast_code_names, index= df.index)
    
    del lane_dmatrix['Intercept'] #delete the intercept column from lane_dmatrix

    #Merge the newly created dummy variables with the original dataframe
    new_df = pd.merge(df, lane_dmatrix, left_index=True, right_index=True)
    
    #Get rid of the original VNUM_LAN column since it is represented by the new
    #Contrast coded VNUM_LAN variables
    del new_df['VNUM_LAN']
    return new_df
    
def clean_category_cols(df):
    """df = a dataframe with columns of categorical variables (and corresponding
    column names) created by patsy's contrast coding functions.
    ==========
    Returns a dataframe where the column names have all been cleaned of the patsy
    syntax. Assumes that any time a reference is present, that it is 9999 to
    signify missing data"""

    cols = df.columns.tolist() #Isolate the column names within df
    r1 = "C(" #The beginning of a contrast coded column name made by patsy
    r2 = ")[" #The ending of contrast coded variable names and beginning of level
    r3 = "]" #The ending of the treatment level delimeter for contrast coded variables
    r4 = ", " #column vs patys option delimiter within the C( )
    r5 = "." #Patsy delimiter before level of original categorical variable
    r6 = "(9999)" #The missing variable treatment delimeter for this specific project
    
    #Create a dictionary where keys = the problematic characters and where
    #values = what the problematic characters should be replaced with
    replace_dict = dict(zip([r1, r4, r2, r5, r3, r6], ["", "_", "_", "", "", "_9999"]))
    
    new_cols = [] #Initialize an empy list to hold the new column names
    for col in cols: #Iterate through the current columns in df
        if r1 in col: #Check if the variabe was made via contrast coding. If so,
            new_col = deepcopy(col) #Make a copy of the column name
            #Iterate through all symbols that will cause the statsmodels formula to malfunction
            for item in replace_dict: 
                #Replace the problematic character with the chosen "fixes"
                new_col = new_col.replace(item, replace_dict[item])
            new_cols.append(new_col) #Place the altered column name in new_cols
        else: #If the column was not a patsy contrast coded variable
            new_cols.append(col) #Place it in new_cols, unaltered
    new_df = df.copy() #Copy the dataframe passed into the function
    new_df.columns = new_cols #Set the columns of the new dataframe
    return new_df  
    
def design_for_hybrid(orig_nodeAugmented, tree, missing_code=9999):
    """orig_nodeAugmented = a pandas dataframe containing a "nodeNum" column.
    orig_nodeAugmented should be the design matrix used to build a decision tree, 
    with the subsequently created nodes used to characterize each observation.
    
    tree = the tree used to characterize records in orig_nodeAugmented
    missing_code = the value used by orig_nodeAugmented to denote missing data
    ==========
    Returns a dataframe which:
       ---only contains records in leaf nodes that are not perfectly predicting
       ---has contrast coded node numbers (one per node)
       ---has cleaned up column names that will work with statsmodels formulas"""
         
    #missing_value_cols = ["TRAV_SP", "NUMOCCS", "VNUM_LAN", "VSPD_LIM", "DRIVER_AGE"]
    
    assert "nodeNum" in orig_nodeAugmented.columns
    
    #Filter the data to avoid perfectly predicting nodes
    filtered = reduce_nodeAugmented_4hybrid(orig_nodeAugmented, tree)
    
    #Contrast code the node numbers in the dataframe
    contrasted = contrast_nodes_in_df(filtered)
    
    #Create appropriate columns for the continuous variables with missing data
    cont_incorporated = incorporate_continuous_vars(contrasted, missing_code)
    
    #Clean the column names so they will be compatible with statsmodels formulas
    col_cleaned = clean_category_cols(cont_incorporated)
    
    filtered = None #Clean up/free up memory
    contrasted = None #Clean up/free up memory
    
    if "nodeNum" in col_cleaned.columns:
        del col_cleaned["nodeNum"] #remove nodeNum col from the design matrix
    if "Intercept" in col_cleaned.columns:
        del col_cleaned["Intercept"] #remove intercept from the design matrix
    
    return col_cleaned
    
def design_for_pure_logit(orig_nodeAugmented, missing_code=9999):
    """orig_nodeAugmented = a pandas dataframe containing a "nodeNum" column.
    orig_nodeAugmented should be the design matrix used to build a decision tree, 
    with the subsequently created nodes used to characterize each observation.
    
    tree = the tree used to characterize records in orig_nodeAugmented
    missing_code = the value used by orig_nodeAugmented to denote missing data
    ==========
    Returns a dataframe which:
    ---has cleaned up column names that will work with statsmodels formulas
    ---has no nodeNum information or columns
    ---created special columns for continous variables that are missing data"""
    
    #missing_value_cols = ["TRAV_SP", "NUMOCCS", "VNUM_LAN", "VSPD_LIM", "DRIVER_AGE"]
    
    #Create appropriate columns for the continuous variables with missing data
    cont_incorporated = incorporate_continuous_vars(orig_nodeAugmented, missing_code)
    
    #Clean the column names so they will be compatible with statsmodels formulas
    col_cleaned = clean_category_cols(cont_incorporated)
    
    del col_cleaned["nodeNum"] #remove the nodeNum col from the design matrix
    if "Intercept" in col_cleaned.columns:
        del col_cleaned["Intercept"] #remove the intercept from the design matrix
        
    return col_cleaned

def isolate_node_cols(df):
    """df = a dataframe containing a set of contrast coded node number columns
    ==========
    Returns a list of all the column names that indicate what node a record
    belongs to."""
    
    col_List = df.columns.tolist() #Get all the columns in the dataframe 
    node_cols = [] #Initialize an empty list for the node number columns
    for item in col_List: #Iterate over all the columns in the dataframe
        if "nodeNum" in item: #If nodeNum is in the column string
            node_cols.append(item) #Add the column string to the list node_cols
    return node_cols
    
def get_independent_cols(df, desired_cols, tol=1e-05):
    """"df = the dataframe containing the data used to build the model.
    desired_cols = a list of column names that we want to use to filter the data.
    ==========
    Returns the largest possible list of list of linearly independent columns, given the
    desired_cols. The solution uses the Q-R matrix decomposition and returns a set of linearly
    independent columns which may not be the only unique set of independent columns."""
    A = df[desired_cols] #Filter the dataframe by the appropriate columns
    Q, R = np.linalg.qr(A) #Calculate the Q and R decompositions of the filtered dataframe
    independent_col_indices = np.where(np.abs(R.diagonal()) > tol)[0] #Isolate the positions of the independent columns
    independent_cols = [] #Initialize an empty list to store the independent columns in
    for i in independent_col_indices: #Fill independent_cols by going through the indices of independent columns
        independent_cols.append(desired_cols[i])
    return independent_cols

def check_full_rank(df, desired_cols):
    """"df = the dataframe containing the data used to build the model.
    desired_cols = a list of column names that we want to use to filter the data.
    ==========
    Returns a boolean saying whether the theoretical rank of the matrix created by filtering
    data by the desired columns is greater than actual, computed rank of the filtered matrix."""
    assert isinstance(desired_cols, list) #Make sure that desired_cols is a list
    for each_col in desired_cols: #Iterate through each column in desired_cols
        assert each_col in df.columns #Check that each column is in the dataframe
    theoretical_rank = len(desired_cols) #Calculate matrix's theoretical rank
    actual_data = df.loc[:,desired_cols] #Filter the data by the desired columns
    actual_rank = np.linalg.matrix_rank(actual_data) #Calculate the matrix's rank
    if actual_rank < theoretical_rank:
        return False
    else:
        return True


def combat_multi_collinearity(df, variables, must_include, max_cond=None):
    """df = the dataframe containing the data that the model will be built on.
    variables = a list of all of the variables to appear in the model or of just the optional independent variables that 
    may be removed from the model
    must_include = a list of independent variables that MUST remain in the model.
    max_cond = a number used to indicate the maximum allowable conditioning number. Will default to 1 / the system floating
    point arithmetic error if not otherwise specified.
    ==========
    Returns the variables that together produce a conditioning number lower than the maximum allowable condition number."""
    assert isinstance(variables, list) #make sure variables and must_include are both lists
    assert isinstance(must_include, list)
    
    #Set the maximum allowable condition number for the matrix
    eps = sys.float_info.epsilon
    ref_cond = 1.0 / eps if max_cond is None else max_cond
    
    #Isolate the variables that are to be added or subtracted from the model to achieve the desired conditioning.
    to_check = list(set(variables).difference(set(must_include)))
    
    if not check_full_rank(df, to_check): #make sure the dataframe of potential variables is of full rank
        to_check = get_independent_cols(df, to_check) #If it is not, remove the appropriate number of linearly dependent variables
    
    tot_vars = to_check + must_include #Create a list of total variables to include in the model, initially
    current_cond_no = np.linalg.cond(df[tot_vars]) #Calculate the conditioning number for those variables
    
    #Remove variables sequentially from to_check while the data's condidtioning is greater than the maximum desired conditioning.
    while current_cond_no > ref_cond:
        to_check, current_cond_no = remove_collinear_var(df, to_check, must_include)
        if len(to_check) == 0: #If no more variables to remove, simply return the variables that must be included in the model
            return must_include
    print "The condition number of the dataset and reduced variable list is {}".format(current_cond_no)
    print "There are {} variables currently in the model.".format(len(to_check + must_include))
    return to_check + must_include

def remove_collinear_var(df, check_vars, include_vars):
    """df = the dataframe containing the data that the model will be built on.
    check_vars = a list of optional independent variables that may be removed from the model
    include_vars = a list of independent variables that MUST remain in the model.
    ==========
    Finds, removes, and returns the variable(s) which, when removed, produce a conditioning number lower than that of the 
    variables passed into the function."""
    
    #Initialize the lowest conditioning number to be the current conditioning number using all desired variables
    lowest_cond_no = np.linalg.cond(df[check_vars + include_vars])
    worst_var = None #Initialize the worst_variable to be none
    new_to_check = deepcopy(check_vars) #Make a copy of check_vars that can be altered without inadvertently changeing check_vars
    
    while worst_var is None: #Keep removing variables until a 'worst_variable' is found, i.e. when the lowest_cond_no is reduced.
        bad_var, cond_no = find_most_collinear_var(df, new_to_check, include_vars) #Find most collinear variable in new_to_check
        if cond_no < lowest_cond_no: #Check if the condition number achieved by removing bad_var is lower than lowest_cond_no
            print "I found a worst variable. The condition number is now {}".format(cond_no)
            lowest_cond_no = cond_no #If so, update the variable assignment for lowest_cond_no and 
            worst_var = bad_var #Assign worst_var to the variable bad_var
        if worst_var is None: #If the condition number achieved by removing bad_var is not lower than lowest_cond_no
            assert bad_var in new_to_check #make sure bad_var is in new_to_check before trying to remove it from new_to_check
            new_to_check.remove(bad_var) #remove bad_var from new_to_check
        else: 
            assert worst_var in new_to_check #Check that worst_var is in the new_to_check list
    
    assert worst_var is not None #Check that worst_var has been assigned to a variable
    new_to_check.remove(worst_var) #Remove worst_var from new_to_check 
    return new_to_check, lowest_cond_no #Return new_to_check along with the newly achieved condition number.

def find_most_collinear_var(df, opt_ind_vars, necessary_vars):
    """df = the dataframe containing the data that the model will be built on.
    opt_ind_vars = a list of optional independent variables that may be removed from the model
    necessary_vars = a list of independent variables that MUST remain in the model.
    ==========
    Finds the variable which, when removed, produces the lowest conditioning number among those in opt_ind_vars"""
    
    lowest_current_cond = np.linalg.cond(df[opt_ind_vars[1:] + necessary_vars]) #Initialize the lowest condition number
    most_collinear = opt_ind_vars[0] #Initialize the variable that produces the lowest condition number
    for var in opt_ind_vars: #Iterate over all of the optinal independent variables
        opt_regressors = deepcopy(opt_ind_vars) #initialize a list of the optional regressors
        opt_regressors.remove(var) #Remove the variable in question
        dataset = df[opt_regressors + necessary_vars] #Filter the dataset to the full set of model variables
        specific_cond_no = np.linalg.cond(dataset) #Calculate the condition number achieved by removing var
        if specific_cond_no < lowest_current_cond: #Check if removing this variable produces the lowest condition number so far 
            most_collinear = var #If so, assign the 'most_collinear' variable name to this variable and
            lowest_current_cond = specific_cond_no #Assign the 'lowest_of_current'_variables name to this variable
    return most_collinear, lowest_current_cond #After iterating through all variables, return most_collinear and lowest_of_current
    
def check_initial_specification(dataframe, result_string, new_var, min_specification, fit_word=None):
    assert isinstance(dataframe, pd.DataFrame) #Make sure dataframe is a pandas dataframe.
    assert isinstance(result_string, str) #Make sure the result_string is actually a string
    assert isinstance(new_var, list) #Make sure new_var is a list
    assert isinstance(min_specification, str) #Make sure the min_specification is a string
    
    base_vars = min_specification.split(" + ") #Extract the variables used in the minimum specification
    if "0" in base_vars: #Remove any zeros from the variables used in the minimum specification
        base_vars.remove("0")
    
    #Initialize starting values for the optimization
    start_vals = np.random.rand(len(base_vars + new_var))
    
    #Create the formula string for the logistic regression
    fString = result_string + " ~ " + min_specification + " + " + " + ".join(new_var)
    
    #Make sure the matrix for the logistic regression is invertible
    if not check_full_rank(dataframe, base_vars + new_var):
        #If not, raise an error
        raise Exception("The base model plus {} is not of full rank.".format(new_var))
    
    #Fit the logistic regression
    if fit_word is None:
        model = smf.logit(fString, data=dataframe).fit(start_params = start_vals, maxiter=2000)
    else:
        model = smf.logit(fString, data=dataframe).fit(method=fit_word, start_params = start_vals, maxiter=2000)
        
    if not model.mle_retvals["converged"]: #Check if the model converged
        #If it did not, raise an error
        raise Exception("The model for {} did not converge".format(new_var))
        
    lowest_pval = model.pvalues[new_var[0]] #Initialize a value for the lowest p-value
    for orig_var in new_var: #Iterate through the new variables
        current_pval = model.pvalues[orig_var]
        #If the current variables p-value is less than the lowest p-value
        if current_pval < lowest_pval:
            #Keep track of this number
            lowest_pval = current_pval
    return lowest_pval

def initial_multivariate(df, res, missing_value=9999, base_string=None, biggest_cond = 1.0e10, fit=None):
    """df = the dataframe containing the data used to build the model.
    res = the string used to name the column of results in df
    missing_value = the number denoting missing values in this dataset
    base_string = the string used to denote the right side of the equation in a patsy formula
    biggest_cond = the maximum value that you will accept for the condition number of a design matrix used in a logit model
    fit = a string used to denote the fit method of the statsmodels logit model
    
    ===========
    
    Returns a fitted model that tried all the variables in the data set and created a multivariate model based
    on all the variables which were significant at a level of less than 0.2 and not linearly dependent. Also
    returns the base string of independent variables, minus the intercept suppressor."""
    
    if base_string is None: #If we are reading the very first variable, then base_string will not be passed in. Instead, create it.
        base_vars = isolate_node_cols(df)
        base_string = " + ".join(["0"] + base_vars) #Make base string from all the node dummy variable columns
    else:
        base_vars = base_string.split(" + ") #Split the base string around plus sign to isolate the variables
        if "0" in base_vars:
            base_vars.remove("0") #Don't include the 0 which tells patsy not to include an intercept in the design matrix
    
    multi_cols = [] #initialize an empty list for the lists of variables that are to remain together
    continuous_vars = ["TRAV_SP", "NUMOCCS", "VSPD_LIM", "DRIVER_AGE"]
    for item in continuous_vars: #Iterate over all the continuous variables and create lists of them and their associated versions
#         multi_cols.append([item, "{}_{}".format(item, missing_value),
#                            "{}_interact_{}_{}".format(item, item, missing_value)]) #Add the entire list of variables to multi_cols
         multi_cols.append([item, "{}_interact_{}_{}".format(item, item, missing_value)])
    
    
    cols = df.columns.tolist() #Initialize the list of columns to add variables from
    for var in base_vars + [res]: #Iterate through the base variables for the model and the dependent variable
        if var in cols: #If the base variable is in cols, remove it
            cols.remove(var) #Make sure cols only has variables to be added to the model, not variables already in the model
    
    print multi_cols
    for var1, var2 in multi_cols: #Iterate through the sets of columns that are to remain together
        #check if each of the variables in the set are in cols, and if they are present, remove them from cols
        if var1 in cols:
            cols.remove(var1)
        if var2 in cols:
            cols.remove(var2)
#         if var3 in cols:
#             cols.remove(var3)
        assert var1 not in cols
        assert var2 not in cols
        #assert var3 not in cols
        
    possible_vars = [] #Initialize an empty variable to keep track of the possible variables for the multivariate model
    problems = [] #Initialize a list to keep track of the variables where the model could not estimate and I don't know why.
    for col in cols + multi_cols: #Iterate over all the variables or groups of variables to be checked
        try:
            #analyze either a list containing a single variable or a list of multiple variables
            to_analyze = [col] if isinstance(col, str) else col
            #get the p-value for the single variable or the most significant p-value from the set of variables
            p_val = check_initial_specification(df, res, to_analyze, base_string, fit_word=fit)
            if p_val < 0.2: #make sure the variable is statistically significant to some loose degree
                if isinstance(col, str):
                    possible_vars.append(col)
        except Exception as inst:
            print inst
            if "did not converge" in str(inst):
                problems.append(col)
        except:
            print "Program encountered an unexpected error on column {}".format(col)
            problems.append(col)
            
    
    print "="*10 #Separate the next line from the printed things above
    print "" #Place an empty line of whitespace
    print "There were estimation problems with at least {} variables. The problematic columns were: ".format(len(problems))
    print problems
    print "="*10 #Separate the line above from the next printed thing below
    print "" #Place an empty line of whitespace
    print "The possible variables being fed to combat_multi_collinearity are: "
    print possible_vars
    
    #Reduce possible_vars to a list of linearly independent variables with a condition number < the specifed condition number
    model_vars = combat_multi_collinearity(df, possible_vars, base_vars, max_cond=biggest_cond)
    new_b_string = " + ".join(model_vars) #Create a string of all variables using in the multivariate regression
    new_fString = res + " ~ " + "0 + " + new_b_string #Create the new formula string
    if fit is None:
        multi_model = smf.logit(new_fString, data = df).fit(maxiter=2000) #fit the initial multi-variate logistic regression
    else:
        multi_model = smf.logit(new_fString, data = df).fit(method=fit, maxiter=2000)
    print multi_model.summary() #Look at the sumary statistics from this logistic regression
    base_string = None #Clean up and reset the variable to none. Not sure if it's necessary but it might not hurt
    return multi_model, new_b_string #return the fitted model and base string used in the model
    
def whittle_multi_model_vars(orig_fitted, base_string):
    """orig_fitted = an object returned from calling .fit() on a statsmodels logit model
    base_string = the right hand side of the formula used to estimate orig_fitted
    ==========
    Returns a list of variables names.
    
    If at least one variable has a p-value which is > 0.05, this function will
    removes the variable with the worst p-value and return a list of all of the
    remaining variables."""
    
    #Create a list of variables in the model using the base_string
    orig_vars = base_string.split(" + ")
    if "0" in orig_vars:
        #Remove the "0", intercept-suppressing term if its present.
        orig_vars.remove("0")
    
    #Get a series of p-values from the fitted model, placed in descending order
    p_vals = orig_fitted.pvalues.order(ascending=False)
    worst_var = None #Initialize a variable for the worst variable from the model
    
    #Iterate through all the variables in the model
    for var, val in p_vals.iteritems():
        if "nodeNum" in var:#Ignore the variables denoting leaf node membership
            continue
        elif "Intercept" == var: #Ignore any variables that are intercept terms
            continue
        else: #If a p-value is greater than 0.05, call it the worst variable
            if val > 0.05:
                worst_var = var
            #Note that the list is in descending order so the first variable
            #identified has the largest p-value of variables which are optional
            break #Break the loop since I have the worst variable

    
    if worst_var is not None:        
    #If a worst variable has been identified, then make a copy of the original
    #variables, remove the worst variable from this copy, and return the rest.
        if np.nan in p_vals.values:
            nan_vars = p_vals[pd.isnull(p_vals)].index.tolist()
            if worst_var not in nan_vars:
                worst_var = nan_vars[0]
                
        new_vars = deepcopy(orig_vars)
        try:
            assert worst_var in new_vars
        except:
            raise AssertionError ("Somehow, {} is not in new_vars (which is {})".format(worst_var, new_vars))
        new_vars.remove(worst_var)
        return new_vars
    else: #If all the variables were significant then return None
        return None
        
def reduce_multi_model(orig_fitted, base_string, res, df, fit=None):
    """orig_fitted = an object returned from calling .fit() on a statsmodels logit model
    base_string = the right hand side of the formula used to estimate orig_fitted
    res = The string for the column name in df that has the classes.
    df = the pandas dataframe from which orig_fitted was estimated
    ==========
    Returns a fitted logistic regression model, and the base string used to estimate
    the model.
    
    If at least one variable has a p-value which is > 0.05, this function will
    removes the variable with the worst p-value, estimate a new logistic regression,
    and repeat the process until no more insignificant variables can be removed."""
    
    #Check the class of the function inputs
    assert isinstance(base_string, str)
    assert isinstance(res, str)
    assert isinstance(df, pd.DataFrame)
    
    #Try to reduce the number of variables in the original model
    new_bvars = whittle_multi_model_vars(orig_fitted, base_string)
    #Initialize a variable for the smallest model
    small_model = orig_fitted
    #Initialize a variable for the smallest model base_string
    small_base = base_string
    
    node_variables = isolate_node_cols(df)
    
    while new_bvars is not None: #If a reduced set of variables has been found
        #new_base = " + ".join(["0"] + new_bvars) #Create a new base_string
        #new_fstring = res + " ~ " + new_base #Create a new statsmodels formula string
        
        model_vars = combat_multi_collinearity(df, new_bvars, node_variables, max_cond=2000)
        new_base = " + ".join(model_vars) #Create a string of all variables using in the multivariate regression
        new_fstring = res + " ~ " + "0 + " + new_base #Create the new formula string
        
        try: #Try to fit a new logistic regression model
        #Use the if...else statement to accomodate various optimization methods
            if fit is None:
                new_model = smf.logit(new_fstring, data = df).fit(maxiter=2000)
            else:
                new_model = smf.logit(new_fstring, data = df).fit(method=fit, maxiter=2000)
        #Assign small_base to the smallest identified set of base variables so far
            small_base = " + ".join(new_bvars)  
        #Assign small_model to the model with smallest set of base variables so far
            small_model = new_model
        #Search for new base variables
            new_bvars =  whittle_multi_model_vars(new_model, new_base)
        except Exception as inst: #If the model could not be fit, print a message saying so
            print "Estimating logit model failed when using formula: {}".format(new_fstring)
            #Note the line below is un-tested, but I added it because it seemed
            #that an infinite loop would result without it.
            print inst
            new_bvars = None

    #Print the model results of the most reduced model.            
    print "="*10
    print "The reduced model results are:"
    print small_model.summary()
    
    return small_model, small_base
    
def calc_error(data):
    """data = a pandas dataframe with two columns, one of class predictions and 
    one of the true classes.
    ==========
    Returns the error rate, defined as the number of incorrect predictions 
    divided by the number of total predictions"""
    
    #Make sure data is a dataframe with two columns
    assert isinstance(data, pd.DataFrame)
    assert len(data.columns) == 2
    
    tot = len(data) #Get a total count of predictions being made
    count_err = 0 #Initialize a running count of the number of wrong predictions
    for ind, row in data.iterrows(): #Iterate over each prediction
        if row.iloc[0] != row.iloc[1]: #Check if the prediction equals reality
            count_err += 1 #If it does not, increment count_err by one
    return float(count_err)/tot #Return the error rate value for data.
      
def hybrid_classify(aug_vSet, tree, fitted, res, classes, fit=None):
    #List the index values in aug_vSet
    tot_ind = aug_vSet.index.tolist()
    #Create a design matrix for the observations to be classified via the hybrid logit model
    logit_design = design_for_hybrid(aug_vSet, tree) 
    #List the index values of observations to be classified by the hybrid logit model
    logit_ind = logit_design.index.tolist()
    #List the index values of observations to be classifed by the decision tree
    tree_ind = list(set(tot_ind).difference(set(logit_ind)))
    
    if len(tree_ind) > 0:
        #Create a design matrix of only observations to be classified by the decision tree
        tree_matrix = aug_vSet.loc[tree_ind]
        tree_preds = [] #Initialize a list to store the decision tree's predictions
        #Iterate over all observations to be classified by the decision tree
        for ind, row in tree_matrix.iterrows():
            #Make the prediction with the decision tree and add it to tree_preds
            tree_preds.append(treepredict.predictWithTree(row, tree, classes))
        #Convert the tree_preds list into a pandas dataframe, with the appropriate index
        tree_preds = pd.DataFrame(tree_preds, columns=["Predictions"], index=tree_ind)
    
    #Isolate the columns needed for the hybrid logit model's design matrix
    cols = fitted.pvalues.index.tolist()
    for col in cols:
        #Iterate through all columns to see if they're in the design matrix
        #Ignore columns that will be automatically created (those with 'I()')
        if col not in logit_design and "I(" not in col:
            #If the column is not in the matrix, such as some node column
            #Make that column be all zeros in the design matrix
            logit_design[col] = 0
    
    #Create a numpy array version of the design matrix for the model
    final_logit_design = np.array(logit_design[cols]) 
    #Get probabilities of being in class 1 from the statsmodel predict() function
    logit_probs = fitted.predict(exog = final_logit_design, transform=False)
    #Get predictions based on the probabilites above
    logit_preds = [1 if x >=0.5 else 0 for x in logit_probs]
    #Turn the list of predictions into a pandas dataframe
    logit_preds = pd.DataFrame(logit_preds, columns=["Predictions"], index= logit_ind)
    
    if len(tree_ind) > 0:
        #Re-combine the decision tree and hybrid logit predictions
        total_preds = pd.concat([logit_preds, tree_preds])
    else:
        total_preds = logit_preds
    #Re-order the combined list of prediction to match the original ordering
    total_preds = total_preds.loc[tot_ind]
    #Isolate the true classes from the observations in the validation set
    true_class = aug_vSet[res]
    
    #Create a 2 column dataframe of the predictions and actual classes
    preds_and_real = total_preds.copy()
    preds_and_real["True Class"] = true_class
    
    #Isolate the observations whose true class is 0 or 1
    class_0 = preds_and_real[preds_and_real['True Class'] == 0]
    class_1 = preds_and_real[preds_and_real['True Class'] == 1]
    
    #Calculate the prediction error rates for each class and overall.
    class_0_err = calc_error(class_0)
    class_1_err = calc_error(class_1)
    tot_err = calc_error(preds_and_real)
    
    cond1 = preds_and_real["Predictions"] == 1
    cond2 = preds_and_real["True Class"] == 1
    true_positive = len(preds_and_real[cond1 & cond2])
    test_positive = len(preds_and_real[cond1])
    actual_positive = len(preds_and_real[cond2])
    
    precision = float(true_positive) / test_positive
    recall = float(true_positive) / actual_positive
    
    #Print resulting error rates and 
    print "When using the hybrid-CART Logit model, the various errors are: "
    print ""
    print "Total Error: {}".format(tot_err)
    print "Class 0 Error Rate: {}".format(class_0_err)
    print "Class 1 Error Rate: {}".format(class_1_err)
    print ""
    print "Note: The number of non-zero logit predictions is {}".format(len(logit_preds[logit_preds["Predictions"] != 0]))
    
    #Return a list of the total, class 0, and class 1 error rates
    return [tot_err, class_0_err, class_1_err, precision, recall]
    
def strip_nodeNum_cols(df):
    """df = a dataframe
    ==========
    Removes the categorized nodeNum columns from the dataframe."""
    assert isinstance(df, pd.DataFrame) #Made sure df is a pandas dataframe
    for var in df.columns.tolist(): #Iterate over all columns of the dataframe
        if "nodeNum" in var: #If the column name contains 'nodeNum'
            del df[var] #Delete it from the dataframe
    return

def logit_error_rates(testSet, model, res_string, the_classes, correct_tree):
    """testSet = a node augmented dataframe on which predictions are to be made
    model = the fitted logistic regression model which will make predictions on data
    res_string = The string for the column name in testSet that has the classes.
    the_classes = a list of classes in the res column to get prediction results for
    correct_tree = the decision tree that this logit model will be compared to
    ==========
    Returns a dictionary of the various error rates for the logistic regression"""
    
    #Create a design matrix for use in testing a pure logit model
    correct_df = design_for_pure_logit(testSet)
    #Get a list of needed columns in the design matrix based on the fitted model
    req_cols = model.pvalues.index.tolist()
    
    for req_col in req_cols:
    #If the column is not a created column and is not in the design matrix: 
        if req_col not in correct_df and "I(" not in req_col:
            correct_df[req_col] = 0 #Initialize the missing column to zero
        #Make sure the column is in correct_df
        assert req_col in correct_df.columns
    
    print "Was about to get predicted probabilities"
    #For each observation, get the probability of it being in class 1
    logit_probabilities = model.predict(exog = np.array(correct_df[req_cols]), transform = False)
    #Predict each observation's class membership based on its probability above
    logit_predictions = pd.DataFrame([1 if x >= 0.5 else 0 for x in logit_probabilities], columns=["Predictions"],
                                     index = correct_df.index)
    
    #Isolate the series of actual class memberships
    actual = testSet[[res_string]].copy()
    actual.columns = ["True Class"]
    
    #Combine the predictions and actual class memberships into a single dataframe
    prediction_and_reality = pd.merge(logit_predictions, actual, left_index = True, right_index = True)
    #Calculate the overall prediction error rate
    tot_err = calc_error(prediction_and_reality) 
    
    #Print out the results and separating characters/lines for readability
    print "="*10
    print ""
    print "When using the pure logit model, its various error rates are:"
    print "Total Error Rate: {}".format(tot_err)
    
    #Calculate the error rates for each individual class
    class_errs = [] #Initialize an empty list for the error rates for each class
    for state in the_classes: #iterate over all classes passed in as important
        #Filter the predictions to only contain observations whose true class = state
        obs_in_state = prediction_and_reality[prediction_and_reality['True Class'] == state]
        #Calculate the prediction error rate for observations whose true class = state
        state_error = calc_error(obs_in_state)
        print "Class {} Error Rate: {}".format(state,state_error) #Print results
        class_errs.append(state_error) #Add the error rate to class_errs
    
    cond1 = prediction_and_reality["Predictions"] == 1
    cond2 = prediction_and_reality["True Class"] == 1
    true_positive = len(prediction_and_reality[cond1 & cond2])
    test_positive = len(prediction_and_reality[cond1])
    actual_positive = len(prediction_and_reality[cond2])
    
    precision = float(true_positive) / test_positive
    recall = float(true_positive) / actual_positive
    
    errs = [tot_err] + class_errs + [precision, recall] #make a list of all error types
    print ""
    print "Note: The number of non-zero logit predictions is {}".format(len(logit_predictions[logit_predictions["Predictions"] == 1]))
    return errs #Return the errors of the pure logit model's predictions.

def predict_with_hybrid_and_tree(data, some_tree, some_model, some_res_string, some_classes):
    """data = a node augmented dataframe on which predictions are to be made
    some_tree = the decision tree that will be used to make predictions on data
    some_model = the hybrid CART-Logit model which will make predictions on data
    some_res_string = The string for the column name in data that has the classes.
    some_classes = a list of classes in the res column to get prediction results for
    ==========
    Returns a dictionary of the various error rates for the hybrid CART-Logit
    model and the decision tree"""
    
    #Get a list of the various error rates as achieved by the hybrid logit model
    hybrid_errs = hybrid_classify(data, some_tree, some_model, some_res_string, some_classes)
    
    #Print separating characters to enhance readability of the various results
    print "="*10
    print ""
    print "="*10
    
    #Get a list of the various error rates as achieved by the decision tree
    tree_errs = treepredict.testAccuracy(data, some_tree, some_classes, some_res_string)
    
    #Combine the errors into a single dataframe
    all_errs = pd.DataFrame({'Hybrid': hybrid_errs,
                             'Tree': tree_errs},
                            index = ['Total Error Rate'] + ['Class {} Error Rate'.format(x) for x in some_classes] + ["Precision", "Recall"])
    return all_errs

def predict_with_hybrid_tree_and_logit(data, some_tree, some_model, some_pure_logit, some_res_string, some_classes):
    """data = a node augmented dataframe on which predictions are to be made
    some_tree = the decision tree that will be used to make predictions on data
    some_model = the hybrid CART-Logit model which will make predictions on data
    some_pure_logit = the fitted logistic regression which will make predictions on data
    some_res_string = The string for the column name in data that has the classes.
    some_classes = a list of classes in the res column to get prediction results for
    ==========
    Returns a dictionary of the various error rates for the hybrid CART-Logit
    model, the pure logistic regression, and the decision tree"""
    
    #Get a list of the various error rates as achieved by the hybrid logit model
    hybrid_errs = hybrid_classify(data, some_tree, some_model, some_res_string, some_classes)
    
    #Print separating characters to enhance readability of the various results
    print "="*10
    print ""
    print "="*10
    
    #Get a list of the various error rates as achieved by the decision tree
    tree_errs = treepredict.testAccuracy(data, some_tree, some_classes, some_res_string)
    
    #Get a list of the various error rates as achieved by the decision tree
    logit_errors = logit_error_rates(data, some_pure_logit, some_res_string, some_classes, some_tree)
    
    #Combine the errors into a single dataframe
    all_errs = pd.DataFrame({'Hybrid': hybrid_errs,
                             'Tree': tree_errs,
                             'Logit': logit_errors},
                            index = ['Total Error Rate'] + ['Class {} Error Rate'.format(x) for x in some_classes] + ["Precision", "Recall"])
                            
    return all_errs
    
def overall_test(aug_tSet, aug_vSet, tree, res, classes, fit_string=None):
    """aug_tSet = a node augmented design matrix for training
    aug_vSet = a node augmented design matrix for validation
    tree = a decision tree built with treepredict.py
    res = a string indicating the column name within aug_tSet and aug_vSet that
    contains the results.
    classes = a list of classes in the res column to get prediction results for
    fit_string = a string indicating the method to use with statsmodels logit
    model class to perform the numerical optimization.
    ===========
    Creates a hybrid CART-logit model, calculates various error rates based on
    both the decision tree and the hybrid CART-Logit model, and returns the
    fitted hybrid model and the dataframe of errors."""
    
    #Create a design matrix to be used when making the hybrid CART-Logit model
    d4logit = design_for_hybrid(aug_tSet, tree)    
    
    #Create an initial hybrid model
    init_multi, init_multi_string = initial_multivariate(d4logit, res, fit=fit_string)
    
    #Get rid of insignificant variables, one at a time
    reduced_multi, reduced_multi_string = reduce_multi_model(init_multi,
                                                            init_multi_string,
                                                            res, d4logit,
                                                            fit=fit_string)
    
    #Calculate the various errors on the validation set, using both the tree and the hybrid model
    errs_dataframe = predict_with_hybrid_and_tree(aug_vSet, tree, reduced_multi, res, classes) 
    
    return reduced_multi, errs_dataframe

def allFeats(fitted, tree):
    """fitted = a statsmodels object returned by calling .fit() on one of the models
    tree = a decision tree created by treepredict.py
    ==========
    Return a set of all the variables used in either the fitted model or the tree.
    
    Note that this set does not contain mutually exclusive variables since a variable
    used in the tree may be the same as one used in the fitted model, but wwritten in
    patsy notation."""
    
    #Make a set of variables from the fitted model
    set1 = set(fitted.params.index.tolist()) 
    #Make a set of variables from the decision tree
    set2 = treepredict.searchFeats(tree)
    return set1.union(set2) #Return the union of the two sets above.
    
def make_logit_models(aug_tSet, res, mod_type, tree=None, fit_term=None):
    if mod_type == "pure":
        design_4logit = design_for_pure_logit(aug_tSet)
    elif mod_type == "hybrid":
        assert tree is not None
        design_4logit = design_for_hybrid(aug_tSet, tree)
    else:
        raise Exception("The model type, mod_type, must be either 'pure' or 'hybrid'. '{}' was entered.".format(mod_type))
    
    initial_logit_model, initial_logit_string = initial_multivariate(design_4logit,
                                                                     res,
                                                                     biggest_cond=2000,
                                                                     fit=fit_term)
    final_logit_model, final__logit_string = reduce_multi_model(initial_logit_model, 
                                                                initial_logit_string,
                                                                res, design_4logit,
                                                                fit=fit_term)
    return final_logit_model
