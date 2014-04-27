import numpy as np, pandas as pd, os

#This variable should be either 11 or 12 for the year 2011 or 2012 respectively
YEAR = 12

def removeDuplicates(df):
    from pandas.util.testing import assert_series_equal
    
    unequalList = [] #Initialize a list to store the duplicate columns with new information
    dups = [] #Initialize a list to store the duplicate columns
    for col in df.columns: #Iterate over all of the columns
        if "_vehDup" in col or "_accDup" in col: #Check if the column is a duplicate column
            dups.append(col) #Add the column name to the list of duplicates
            orig = col[:-7] #Isolate the original column name of the duplicate
            try: assert_series_equal(df[col], df[orig]) #See if the duplicate column and the original column are equal
            except:
                unequalList.append(col) #If the original and duplicate column are not equal, add the duplicate to the unequal list
    equalDups = list(set(dups).difference(set(unequalList))) #Isolate the duplicate columns which are equal
    df.drop(equalDups, axis = 1, inplace = True) #Drop the equal duplicate columns from df. Note that here axis = 1 for columns
    return

def makeNewColumns(df, columnList, initialVals):
    """df = the dataframe to be altered.
    columnList = a list of the new columns to be added to the dataframe
    initialVals = a list of the initial values for the new columns.
    initialVals should be of length 1 or of length == len(columnList).
    If len(initialVals) == 1 then all columns will have the same initial value."""
    
    if len(initialVals) == len(columnList): #Create the tuples of column and inital value pairs
        iterable = zip(columnList,initialVals)
    elif len(initialVals) == 1: #Create the tuples of column and inital value pairs
        val = initialVals[0] #Use the sole value in initialVals.
        iterable = [(x, val) for x in columnList]
    for i, j in iterable:
        df[i] = j #Create and initialize the new columns
    return

def fillNewColumns(ser, people):
    """ser = a row series in the data frame the new column is being made for
    people = the dataframe of the GES person data
    
    ========
    Fills in the new columns of ser using the people dataframe.
    Note this function is "hard-coded" given the columns created with makeNewColumns()."""
    relPpl = persons[persons.CASENUM == ser["CASENUM"]] #Filter the person file to only those involved in the relevant crash
    strikingDriver = relPpl[(relPpl.VEH_NO == ser["STR_VEH"]) & (relPpl.PER_TYP == 1)] #Isolate the driver who struck the cyclist
    if len(strikingDriver) > 0: #Make sure the vehicle has a driver. One record does not have a driver.
        if strikingDriver["AGE"].iloc[0] in [998, 999]: #Check if the age is unknown
            ser["DRIVER_AGE"] = strikingDriver["AGE_IM"].iloc[0] #If the age is unknown, use the imputed age
        else:
            ser["DRIVER_AGE"] = strikingDriver["AGE"].iloc[0] #Get the striking driver's age
        if strikingDriver["SEX"].iloc[0] in [8, 9]: #Check if the sex is unknown
            ser["DRIVER_SEX"] = strikingDriver["SEX_IM"].iloc[0] #If the sex is unknown, use the imputed sex
        else:
            ser["DRIVER_SEX"] = strikingDriver["SEX"].iloc[0] #Get the striking driver's sex
        if strikingDriver["DRUGS"].iloc[0] not in [8, 9]: #Check if the striking driver's drug use is known
            ser["DRIVER_DRUGS"] = strikingDriver["DRUGS"].iloc[0] #If so, use the drug report
    else:
        ser["DRIVER_AGE"] = 9999 #Say you don't know the driver's age
        ser["DRIVER_SEX"] = 9999 #Say you don't know the driver's sex
        ser["DRIVER_DRUGS"] = 9999 #Say you don't know the driver's drug status
    return ser

def calcEntropy(df, col):
    """df = the dataframe the entropy will be calculated based on
    col = the column that the entropy calculation will be based on"""
    total = len(df.index) #Get the total number of rows in the dataframe
    counts = df.groupby(col).size() #Get a dictionary of counts per value for the given column in df.
    probs = [float(x)/total for x in counts.values] #Calculate probabilities for each value in the dataframe
    log2 = lambda x: np.log(x)/np.log(2) #Create a function to calculate log-base-2 of a number
    tot = 0 #Initialize the entropy value
    for prob in probs:
        tot += prob * log2(prob) #Iterate over each value in the column, adding p * log_2(p) to the total
    return -1 * tot #Multiply the total value by negative one to get the entropy value

def removeUnhelpfulColumns(df, filepath, vLimit):
    """df = the dataframe to remove unhelpful columns from
    filepath = a path to a csv file with columns: "Cols" and "Include" with a 1 in "Include for columns to include 
    and 0 for columns to exclude
    vLimit = the lowest acceptable number of observations acceptable in a count of matching observations for a 
    binary variable.
    
    ==========
    This function Will first remove columns manually decided to be unhelpful. Then, it will go through the remaining columns
    and check if the entropy in the column is lower than that in a reference case based on vLimit. Essentially, this removes
    columns with low variability within it."""
    usefulVars = pd.read_csv(filepath) #Read in the csv file which denotes the variables I do and do not want to include
    drop = [] #Initialize a variable to hold the variables that will be dropped from the dataframe.
    for ind, row in usefulVars.iterrows(): #Iterate over all the columns, adding the ones to be dropped to 'drop'
        if row["Include"] == 0:
            drop.append(row["Cols"])
    df.drop(drop, axis = 1, inplace = True) #Drop the columns which I specifed to drop
    log2 = lambda x: np.log(x)/np.log(2) #Create a function to take log_2 of a number
    refProbs = [float(vLimit) / len(df), (len(df) - float(vLimit))/len(df)] #Create probabilities for a minimum entropy score
    refEntropy = 0 #Initialize the minimum entropy
    for p in refProbs:
        refEntropy -= p * log2(p) #Iteratively calculate the minimum entropy to which all columns will be compared.
    drop = [] #Re-initialize an empty list for the column to be dropped
    for col in df.columns: #Iterate over every column in the dataframe
        if calcEntropy(df, col) < refEntropy: #Compare the entropy within the column to the minimum entropy value
            drop.append(col) #If the entropy of the column < the minimum entropy, add it to the list to be dropped from df.
    df.drop(drop, axis = 1, inplace = True) #Drop the columns with too low of an entropy score.
    return

def addSafetyEQ(df, safetyDf):
    """df = the dataframe the safety columns are to be added to
    safetyDf = the dataframe containing the GES safety equipment data
    
    =========
    Returns the modified df"""
    
    cols = ["MSAFEQMT{}".format(x) for x in np.sort(safetyDf.MSAFEQMT.unique().tolist())] #Create the new column names
    makeNewColumns(df, cols, [0]) #Add the new columns to the dataframe and initialize them with zero
    df = df.apply(fillSafetyColumns, axis = 1, args = (safetyDf, cols)) #fill in the new columns
    return df

def fillSafetyColumns(ser, safetyDf, cols):
    """ser = a row from the dataframe having its safety columns added and populated.
    safetyDf = the dataframe containing the GES safety equipment data
    cols = the list of column names for the safety columns"""
    
    records = safetyDf[(safetyDf.CASENUM == ser["CASENUM"]) & (safetyDf.PER_NO == ser["PER_NO"])] #Isolate the relevant records
    positive = records.MSAFEQMT.unique().tolist() #Find out which safety equipment codes are associated with this person
    for col in cols: #Iterate through the safety columns
        if int(col[-1]) in positive: #Check if this column's safety equipment code is associated with this person
            ser[col] = 1 #If so, change this column's value to 1.
    return ser

if __name__ == "__main__":
    accidents = pd.read_table(os.path.join("GES_{}".format(YEAR), "ACCIDENT.TXT"))
    vehicles = pd.read_table(os.path.join("GES_{}".format(YEAR), "VEHICLE.TXT"))
    persons = pd.read_table(os.path.join("GES_{}".format(YEAR), "PERSON.TXT"))
    safety_eq = pd.read_table(os.path.join("GES_{}".format(YEAR), "SAFETYEQ.TXT"))

    cond1 = persons.PER_TYP == 6 #person type == "bicyclist"
    cond2 = persons.PER_TYP == 7 #person type == "other cyclist"
    cond3 = persons.INJ_SEV.isin([2,3,4]) #injury severity is evident and non-incapacitating, incapacitating, or fatal
    cond4 = persons.INJSEV_IM.isin([2,3,4]) #imputed injury severity is evident and non-incapacitating, incapacitating, or fatal
    cyclists = persons[(cond1 | cond2) & (cond3 | cond4)] #cyclists with evident&non-incapacitating, incapacitating, or fatal

    #Augment the person data on the cyclists with the vehicle data for the vehicle which struck the cyclist
    vehMerged = pd.merge(cyclists, vehicles, how="left", left_on = ["CASENUM", "STR_VEH"],
            right_on = ["CASENUM", "VEH_NO"], suffixes = ("", "_vehDup"))

    #Augment the person and vehicle data with accident data
    fullyMerged = pd.merge(vehMerged, accidents, how = "left", on = "CASENUM", suffixes = ("", "_accDup"))

    removeDuplicates(fullyMerged) #Remove duplicate columns which contain no new information

    newCols = ["DRIVER_AGE", "DRIVER_SEX", "DRIVER_DRUGS"] #Create a list of the new columns to be added to fullyMerged

    makeNewColumns(fullyMerged, newCols, [9999]) #Initialize the new columns in fullyMerged

    fullyMerged = fullyMerged.apply(fillNewColumns, axis = 1, args=(persons,)) #Fill in the new columns in fullyMerged

    #Output a csv, which I then use to get the column names in one column. Next to it I place a column called "Include" which contains
    #0 if I don't think the field is useful and 1 if I do think the field is useful. I saved that csv as "fullyMerged_usefulVars.csv"
    fullyMerged.to_csv("fullyMerged_20{}.csv".format(YEAR))

    removeUnhelpfulColumns(fullyMerged, "fullyMerged_usefulVars.csv", 10)

    print fullyMerged.info()

    allButCategorySplit = addSafetyEQ(fullyMerged, safety_eq)

    allButCategorySplit.to_csv("allButCategorySplit_20{}.csv".format(YEAR))
