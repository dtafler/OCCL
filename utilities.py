import os
import pandas as pd
from itertools import combinations, product
from pyswip import Prolog
import cv2
import numpy as np

def decode_kf_classification(objectCode: int):
    """
    decode a kandinsky-figure classification.
    
    Parameters:
        objectCode: the classification as an integer between 0 and 35
        
    Returns:
        color: color of the figure
        small: shape of the figure
        large: shape of the large figure is a part of
    """
    assert objectCode < 36, f"objectCode should be smaller than 36 (is {objectCode})" 
    # check large shape
    if objectCode < 9:
        large = "square"
    elif objectCode < 18:
        large = "circle"
        objectCode -= 9 
    elif objectCode < 27:
        large = "triangle"
        objectCode -= 18
    else:
        large = "no_shape"
        objectCode -= 27
    
    assert objectCode < 9, "objectCode should be smaller than 9"

    # ceck color
    if objectCode < 3:
        color = "red"
    elif objectCode < 6:
        color = "yellow"
        objectCode -= 3
    else:
        color = "blue"   
        objectCode -= 6

    assert objectCode < 3, "objectCode should be smaller than 3"

    # check small shape
    switch = {
        0:"square",
        1:"circle",
        2:"triangle"
    }
    small = switch.get(objectCode)

    return color, small, large

def convert_yolo_output_to_partof_facts(directory, concept=None):
    """
    Read directory with yolo output and convert to prolog facts consisting of "part_of"-predicates.

    Parameters:
        directory: directory that contains numbered .txt files of yolo output
        concept: optional description of the examples' concept (e.g. true, false, counterfactual), default=None

    Returns:
        string with Prolog facts
    """

    facts = ""
    with os.scandir(directory) as it:
        for f in it:
            fName = concept + "_" + f.name[:-4] if concept != None else f.name[:-4]
            with open(f.path, 'r') as kf:
                for line in kf:
                    lineElems = line.split(' ')
                    objectCode = int(lineElems[0])
                    color, smallShape, largeShape = decode_kf_classification(objectCode)
                    x = lineElems[1]
                    y = lineElems[2]
                    w = lineElems[3]
                    h = lineElems[4]
                    # facts += number_to_fact(fName, int(lineElems[0]), lineElems[1], lineElems[2], lineElems[3], lineElems[4]) + "\n"
                    facts += f"part_of(kf_{fName}, {color}, {smallShape}, {largeShape}, {x}, {y}, {w}, {h[:-2]})." + "\n"
    return facts

def get_ground_truths_from_dir_structure(directory):
    """
    Make prolog-fact-ground-truths based on directory structure of training images.
    
    Parameters:
        directory: directory of training images of form mainDir/(true or false or counterfactual)/image-name

    Returns:
        string with prolog facts in the form of pos(kp(kf_t_image-name)), neg(kp(kf_cf_image-name)), or neg(kp(kf_f_image-name))
    """
    exs = ""
    with os.scandir(directory) as mainDir:
        for subDir in mainDir:
            with os.scandir(subDir) as files:
                for file in files:
                    if subDir.name == "true":
                        exs += f"pos(kp(kf_t_{file.name[:-4]})).\n"
                    elif subDir.name == "counterfactual":
                        exs += f"neg(kp(kf_cf_{file.name[:-4]})).\n"
                    else:
                        exs += f"neg(kp(kf_f_{file.name[:-4]})).\n"
    return exs

def get_all_but_first_args(pred: str):
    """
    Get all but the first argument of a prolog predicate as a list
    
    Parameters:
        pred: Prolog predicate in form predicate(arg1, arg2,..., argn) as string
    
    Returns: 
        list with all but the first arguments of the predicate
    """
    sub = pred[pred.find(",")+1: pred.find(")")].strip()
    sub = sub.replace(" ", "")
    return sub.split(",")

def make_dataframe_from_prolog_facts(facts):
    """
    Make dataframe from prolog facts, whereby first argument of each is ignored 
    (as currently only one image is processed at a time and the first argument is the name of the image, it can be ignored).

    Parameters:
        facts: a string of grounded versions of the same prolog facts of the form "p(a1,a2,...an).\np(...)\n..." 

    Returns:
        pandas dataframe with argument-position of prolog-facts as columns
    """
    l = facts.split("\n") # list of predicates
    l = [get_all_but_first_args(pred) for pred in l if pred != ''] # elements consist of list of predicate arguments (except first)
    cols = ["arg" + str(i) for i in range(1, len(l[0])+1)] # create list of column names 
    return pd.DataFrame(l, columns=cols) 

def get_path_to_newest_yolo_result(directory: str):
    """
    Get path to newest result of yolo output from a specified directory.

    Parameters:
        directory: path as string to directory with yolo results

    Returns:
        path as string to newest results (path to a text file)
    """
    files = os.listdir(directory) # list of filenames in format [exp, exp2, exp3, ...]. we need the most recent exp (highest number)
    files = [int(elem[3:]) if len(elem)>3 else 0 for elem in files] # convert to list of ints
    if len(files) == 1: 
        return f"{directory}/exp/labels"
    else: 
        return f"{directory}/exp{str(max(files))}/labels"

def get_combos(lst):
    """
    Get all possible combinations without replacement of the elements of a list.

    Parameters:
        lst: a list 

    Returns: 
        list of tuples; combinations of length 1 to n without replacement of a list of length n; sorted according to length of combination
    """
    combos = []
    for i in range(len(lst)):
        combos.extend(list(combinations(lst, i+1))) # combinations of length i+1 without replacement
    return combos

def get_filter_combos(df): 
    """
    Get cartesian product of unique column-values of a pandas dataframe (last four columns are ignored).
    
    Parameters:
        df: Pandas dataframe 
    
    Returns:
        cartesian product of unique column-values of df (last four columns are ignored) (type: itertools.product; each element a tuple containing a tuple of values for each column of dataframe (except last four columns))
    """
    unique = []
    for col in df.drop(df.columns[[-1,-2,-3,-4]], axis=1): # drop location information
        unique.append(list(df[col].unique())) # get unique values of each column --> each element in unique is a list corresponding to the the unique values of one column
    combos = [get_combos(x) for x in unique] # possible filter combinations for each column (list of lists (columns) of tuples)
    return product(*combos) # combine filter combinations of columns (cartesian product); unpack combos for product to work as intended

def subset_df(df, filter):
    """
    Filter a DataFrame to contain only values specified in filter

    Parameters:
        df: Pandas DataFrame 
        filter: a tuple of tuples, each of which specifies the values to filter for in each column (place-coded)
    
    Returns:
        Pandas DataFrame with values specified in filter. Original DataFrame is not changed
    """
    for i in range(len(filter)):
        dis = pd.DataFrame()
        for val in filter[i]: # disjunction: each column might have multiple values
            dis = pd.concat([dis, df[df.iloc[:,i] == val]])
        df = dis # conjunction: conditions must apply for all columns simultaneously --> values that have been filtered out cannot come back
    return df

def get_distance(df=pd.DataFrame, df2=pd.DataFrame, weights={'arg1':1, 'arg2':1, 'arg3':1})-> float:
    """
    Get distance between two Pandas DataFrames of same number of rows

    Parameters:
        df: Pandas DataFrame
        df2: Pandas DataFrame, should have same column names as df
        weights: dictionary of form {'column-name': int or float} specifying the relative weight of the columns

    Returns:
        float representing a measure of distance between the two DataFrames
    """
    assert len(df) == len(df2), "argument dfs should have equal length" # distance function currently does not take differences in the number of objects into account 

    # create DataFrame diffDF consisting of columns where differences were found and 1s in the rows where these differences are and 0s otherwise
    diffDF = df.drop(df.columns[[-1,-2,-3,-4]], axis=1).compare(df2.drop(df2.columns[[-1,-2,-3,-4]], axis=1)).notnull().astype(int)
    
    # iterate over columns and sum the weighted differences
    x = 0
    for name, values in diffDF.iteritems():
        if name[1] == 'self':
            s = values.sum()
            w = weights.get(name[0]) 
            x += s*w
    return float(x)

def classify(input_df: pd.DataFrame, classifier: str):
    """
    Classify Yolo-Output according to prolog rule.

    Parameters:
        input_df: Pandas dataframe consisting of Yolo-Outputs of one image that have been converted to part_of facts
        classifier: path to a text file containing the prolog rule to be used as a classifier

    Return:
        whether the input_df belongs to the concept specified in the classifier
    """
    # print('START CLASSIFICATION FUNCTION')
    # t1 = time.perf_counter()
    
    with open('classification.pl', 'w') as clsdoc:
        for row in input_df.itertuples(): # convert input_df to prolog facts
            s = "part_of(x"
            for i in range(1, len(row)):
                s += f", {row[i]}"
            s += ").\n"
            clsdoc.write(s)
        
        with open(classifier, 'r') as r: # add classification rule from file 
            rule = r.read()
        rule = rule + "\n"
        clsdoc.write(rule) 
        
        with open('bk-base.pl', 'r') as bk: # add background knowledge 
            clsdoc.write(bk.read()) 
            
    # print('START Consult PL FILE')
    # t2 = time.perf_counter()
    p = Prolog()
    p.consult("classification.pl") 
    # print('START QUERY')
    # t3 = time.perf_counter()
    result = list(p.query('kp(x)')) # evaluate kp(input_df) --> list of dicts with variables and their instantiations
    # print('END CLASSIFICATION')
    # t4 = time.perf_counter()
    # print(f"writing pl file statements: {t2-t1}\n")  
    # print(f"consulting pl file statements: {t3-t2}\n")  
    # print(f"querying: {t4-t3}\n")  
    return len(result) > 0 # if prolog finds a solution, the list of results is not empty

def calculate_near_miss_hit_2(inputDF: pd.DataFrame, combos: product, changesDict: dict, classifier: str):
    """
    Calculate a near miss or near hit using the number of changed small objects as measure of distance; this allows distance to be calculated before applying the changes and thus eliminates the need to apply changes to every subset of objects defined in the filter-combinations; changes are made to only one column at a time.

    Parameters:
        inputDF: input instance as Data Frame
        combos: filter combinations of type itertools.product (each element a tuple containing a tuple of values for each column of dataframe (except last four columns)); define extent and location of changes in input; get combos for instance by calling get_filtered_combos(input) 
        changesDict: dictionary {'column name': [list of possible values]}
        classifier: path to a text file containing the prolog rule to be used as a classifier

    Return:
        near miss/hit 'z' and rows of z that have been modified as DataFrames, or None if no Near Miss/Hit has been found
    """
    inputClass = classify(inputDF, classifier)
    # sort results of filter combinations according to size and start with smallest, except zero
    comboSizes = {}
    for comb in combos:
        inputSubset = subset_df(inputDF, comb)
        comboSizes[comb] = (inputSubset, inputSubset.shape[0])
        # print(f'comb: {comb}')
        # print(f'filteredInput: {filteredInput}')
        # print(f'filteredInput.shape[0]: {filteredInput.shape[0]}')
        # print(f'comboSizes[comb]: {comboSizes[comb]}\n\n')

    sortedComboSizes = {key: val for key, val in sorted(comboSizes.items(), key = lambda x: x[1][1]) if val[1] != 0} # sort according to value and drop 0s
    # for key in sortedComboSizes:
    #     print(f'key: {key}\nsortedComboSizes[key]: \n{sortedComboSizes[key]}\n\n')
    
    for comboKey in sortedComboSizes:
        # apply first change -> classify
        # if different class -> break and search for even smaller change
        # else -> try different changes
       
        for columnName in changesDict:
            
            for singleChange in changesDict[columnName]:
                z = inputDF.copy()
                indecesToChange = sortedComboSizes[comboKey][0].index
                z.loc[indecesToChange, columnName] = singleChange
                zClass = classify(z, classifier)
                
                if zClass != inputClass:
                    return search_for_smaller_change(inputDF=inputDF, inputClass=inputClass, indecesChanged=indecesToChange, columnName=columnName, appliedChange=singleChange, classifier=classifier)
    
    print("Warning: no Near Miss or Near Hit found. Maybe more than one parameter/column needs to be changed.")  
    return None       

def search_for_smaller_change(inputDF, inputClass, indecesChanged, columnName, appliedChange, classifier):
    """
    Search for a smaller change that is also a near hit/miss given.
    
    Parameters:
        inputDF: original input instance as DataFrame
        inputClass: classification result of that input
        indecesChanged: indeces of the rows of the inputDF have been changed to get a near hit/miss
        columnName: name of the column of the input DataFrame where the change was applied
        classifier: path to a text file containing the prolog rule to be used as a classifier
    
    Return: 
        near miss/hit 'z' and rows of z that have been modified as DataFrames
    """
    for toChange in get_combos(indecesChanged):
        z = inputDF.copy()
        z.loc[toChange, columnName] = appliedChange
        zClass = classify(z, classifier)
        if zClass != inputClass:
            toChange = list(toChange)
            return z, z.loc[toChange]


def calculate_near_miss_hit(inputDF: pd.DataFrame, combos: product, changesDict: dict, distWeights: dict, classifier: str):
    """
    Calculate a near miss or near hit by making all possible changes defined in changesDict to those parts of the input defined in combos, calculating the distance for each change and finding the change with the smallest distance that changes the classification of input; changes are made to only one column at a time.

    Parameters:
        inputDF: input instance as Data Frame
        combos: filter combinations of type itertools.product (each element a tuple containing a tuple of values for each column of dataframe (except last four columns)); define extent and location of changes in input; get combos for instance by calling get_filtered_combos(input) 
        changesDict: dictionary {'column name': [list of possible values]}
        distWeights: dictionary {'column name': int} to individually weight columns
        classifier: path to a text file containing the prolog rule to be used as a classifier

    Return:
        near miss/hit 'z' and rows of z that have been modified as DataFrames, or None if no Near Miss/Hit has been found
    """
    inputClass = classify(inputDF, classifier)
    nearMissCandidates = {} # keys are tuples (filter-combination to identify rows to be changed, name of columns to be changed, value to change selected cells to), value are distances of changed DataFrames to input
    
    for comb in combos: 
        inputSubset = subset_df(inputDF, comb) # rows of the input that will be changed
        
        for columnName in changesDict:
            
            for change in changesDict[columnName]: # all permissible changes in dictionary changesDict (values to which entries in this column can be changed)
                z = inputDF.copy()
                z.loc[inputSubset.index, columnName] = change
                dist = get_distance(inputDF, z, distWeights)
                if dist > 0: # add z to the dictionary 'distances' if z is different from input according to distance function and class of z is different to class of y --> candidate for near miss/hit
                    class_z = classify(z, classifier) 
                    if class_z != inputClass:
                        nearMissCandidates[(comb, columnName, change)] = dist    
    
    if not nearMissCandidates:
        print("Warning: no Near Miss or Near Hit found!")
        return None
    
    minDistance = min(nearMissCandidates.values())
    nearMissElem = [key for key in nearMissCandidates if nearMissCandidates[key] == minDistance][0] # find near miss candidate with shortest distance to input
    indecesChanged = subset_df(inputDF, nearMissElem[0]).index
    columnName = nearMissElem[1]
    change = nearMissElem[2]
    return search_for_smaller_change(inputDF, inputClass, indecesChanged, columnName, change, classifier)

def draw_bb(img, coords):
    """
    Draws bounding boxes.
    
    Parameters:
        img: image to draw to
        coords: list with bounding box coordinates [(label,x,y,w,h), ...]
    
    Returns:
        image with bounding boxes drawn in
    """
    for coord in coords:
        x_center = float(coord[1]) * img.shape[1]
        y_center = float(coord[2]) * img.shape[0]
        x1 = int(x_center - (0.5 * (float(coord[3]) * img.shape[1])))
        y1 = int(y_center - (0.5 * (float(coord[4]) * img.shape[0])))
        x2 = int(x_center + (0.5 * (float(coord[3]) * img.shape[1])))
        y2 = int(y_center + (0.5 * (float(coord[4]) * img.shape[0])))

        label = str(coord[0])
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        color = (0,200,0)
        text_color = (0,0,0)

        img2 = np.zeros(img.shape, np.uint8)   
        img2 = cv2.rectangle(img2, (x1,y1), (x2,y2), color, 2) 
        img2 = cv2.rectangle(img2, (x1, y1 - 20), (x1 + w, y1), color, -1)
        img2 = cv2.putText(img2, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        img = cv2.addWeighted(img, 1.0, img2, 0.98, 1)
        # img = cv2.putText(img, label, (x2+1,y1-1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    return img

def concat_labels(lst: list)-> list:
    """
    Concatenate all but the last four elements of a list of strings into the first element (str) of a new list.append

    Parameters: 
        lst: List with strings a elements, must have more than 4 elements
    
    Returns:
        list that is similar to lst, but the all but the last four elements have been concatenated to a single str

    """
    assert len(lst) > 4, f"lst must be have length > 4"
    label = lst[0]
    for i in range(1, len(lst)-4):
        label += ', ' + lst[i]
    return [label] + lst[-4:]