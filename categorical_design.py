import numpy as np, pandas as pd
import patsy
import treepredict


MISSING_VALUE = 9999

def reset_col_types(df, ignore_cols):
    """df = the dataframe whose columns are to be reset
    ignore_cols = a list of column names to ignore when setting the column types
    Sets all the type of all the columns in df, except for the columns in
    ignore_cols, to the type int."""
    #Initialize a list of columns to have their type reset to int.
    cols = df.columns.tolist() #Get a list of all the columns in the dataframe
    #Check if any columns should not have their type reset to int
    if len(ignore_cols) != 0:
        for item in ignore_cols:
    #if so, remove that column from the columns that will have their type reset.
            cols.remove(item)  
    for col in cols: #Iterate over the columns getting their type reset
        df[col] = df[col].astype(int) #Set the column type to int
    return
    
def read_and_reset_col_types(path, ind_col, ignore_cols):
    df = pd.read_csv(path, index_col = ind_col)
    reset_col_types(df, ignore_cols)
    return df

def convertMissing(value):
    """value = a string
    The function determines if value is a number or a 'NaN'.
    ========== 
    Returns either the appropriate number or a numpy not a number object."""
    assert type(value) == str #make sure value is a string
    if value.isdigit(): #Check if all a characters in value are numeric
        return int(value) #If so, return value's correct integer representation
    elif value == "NaN": #If value is NaN, return the corresponding numpy object
        return np.nan
    elif "9" is value: #Check if '9' is value, if so return 9.
        return 9 #I have no idea why I kept getting errors on the TOW_VEH_vehDup column and had to do this.
    else:
        print "You have an error with the value passed to convertMissing: {}".format(value)
        print "It is {} that this is a digit".format(value.isdigit())
        return
        
def parseMissing(origVal, col, columnDf):
    if col not in columnDf.Cols.values:
        return origVal
    missing = columnDf[columnDf.Cols == col].iloc[0]["DontKnow"]
    if missing == "" or pd.isnull(missing): #Check if this column allows missing values in the first place
        return origVal #If not, return the original value
    elif pd.isnull(origVal): #If the original value is null, return 9999
        return MISSING_VALUE
    else:
        missing = missing.split("_") #Make a list from the entries in missing
        missing = [convertMissing(x) for x in missing] #Convert all entries to an appropriate value
        if origVal in missing: #See if the originalValue is code for a missing value
            return 9999 #If so, return 9999
        else: #If the original value is not in missing, return the original value.
            return origVal

def removeMissing(ser, columnDf):
    for col in ser.index: #Iterate through every column for this observation
        ser[col] = parseMissing(ser[col], col, columnDf) #Check each column and convert all missing values to 9999
    return ser
    
def make_design_string(df, columnDf):
    """df = the dataframe the design string for patsy is to be made from.
    df should be the dataframe without missing values
    columnDf = the dataframe describing which variables should be included 
    in the design matrix and which variables are categorical.
    ==========
    Returns a string to be used to create the patsy design matrix"""
    dStringList = [] #Initialize an empty list to hold the terms that will make up the patsy formula for the design matrix
    for col in df: #Iterate over each column in basicTransform
        if col in columnDf.index:
            if columnDf.loc[col,"4Tree"] == 1: #Check if the column is one which I want to include in the decision tree
                if columnDf.loc[col, "Categorical"] == 1: #Check if the column is a categorical variable.
                    if 9999 in df[col].values: #If the column has missing values, make that the reference.
                        #Add the patsy string for the above conditions to the list
                        dStringList.append("C({}, Treatment(9999))".format(col)) 
                    else:
                        if "MSAFE" not in col: #Account for the fact that MSAFEQMT columns have already been separated by categories
                            #If not a MSAFEQMT column, separate the categorical variable in patsy
                            dStringList.append("C({})".format(col))
                        else:
                            #If it's an MSAFEQMT column, add it as is
                            dStringList.append("{}".format(col))
                elif columnDf.loc[col, "Categorical"] == 0: #If the column is not categorical, add it directly to the patsy formula
                    dStringList.append("{}".format(col))
    dStringList.remove("C(INJ_SEV)") #Don't use injury severity, use a 0-1 variable that represents serious injury or not, SER_INJ
    dStringList.remove("MSAFEQMT8") #Make "Not Reported" and "Unknown if Used" the 'unknown'/9999 reference category.
    dStringList.remove("MSAFEQMT9") #Make "Not Reported" and "Unknown if Used" the 'unknown'/9999 reference category.
    dStringList.append("SER_INJ") #Add SER_INJ to the list of variables to be used in the patsy design matrix
    dString = " + ".join(dStringList) #Create the patsy formula string
    return dString
    
def add_injury_indicator(col_name, df):
    """col_name = desired name for the serious injury indicator variable
    df = dataframe with an injury severity column, "INJ_SEV", of values 2,3,or 4
    Adds a column of the name: col_name to the dataframe df."""
    try:
        #Check that col_name is not already an existing column
        assert col_name not in df.columns.tolist()
    except:
        print "col_name already exists as a column name within your dataframe. Please choose a different name."
        return
    df[col_name] = 0 #initialize the new injury indicator variable to be zero
    for ind, row in df.iterrows(): #Iterate over the rows of the dataframe
    #Check if the row corresponds to a person suffering an incapacitating or fatal injury
        if row["INJ_SEV"] in [3,4]: 
            df.loc[ind, col_name] = 1 #If so, change the indicator variable to 0
    return
    
def create_design_df(design_string, df):
    """design_string = the string that tells patsy how to create the design matrix
    df = the dataframe from which the design matrix should be created
    ==========
    Returns a dataframe created from the patsy design matrix that was created by using the
    design string and df."""
    dmatrix = patsy.dmatrix(design_string, data = df) #Create the design matrix
    #Create the dataframe from dmatrix. note, dtype = int since all of the values in the
    #NASS GES that are being used in the design matrix are intetgers (note WEIGHT and the 
    #alcohol or drug tests are not included in the desing matrix).
    designDF = pd.DataFrame(dmatrix, columns = dmatrix.design_info.column_names, dtype = int)
    return designDF
    
def add_node_column(df, col_name, tree):
    """df = the design matrix without the column for node identifying values
    col_name = the name of the column that the node numbers will be placed in.
    tree = the tree containing the leaf nodes that will be used to characterize the dataset
    ==========
    Returns a new dataframe that is a copy of df with a column of identifying node numbers"""
    new_df = df.copy() #Copy df
    tree_nodes = treepredict.fetchNodes(tree) #Get a list of nodes for the given tree
    new_df[col_name] = np.nan #Initialize the column of node numbers
    #Fill in the new column with the correct node numbers
    new_df = new_df.apply(treepredict.set_node_num, axis = 1, args = (col_name, tree_nodes))
    return new_df
    