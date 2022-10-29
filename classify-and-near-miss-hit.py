from utilities import calculate_near_miss_hit_2, convert_yolo_output_to_partof_facts, make_dataframe_from_prolog_facts, get_path_to_newest_yolo_result, classify, get_filter_combos, calculate_near_miss_hit, concat_labels, draw_bb
import argparse
from yolov5 import detect
import cv2

def run(weights, input, classifier): # currently processes single images as input

    ####################################
    #### segment and classify input ####
    ####################################
    print('\n\n#####################\n\nDetecting Objects\n\n#####################')
    detect.run(weights=weights, source=input, nosave=True, save_txt=True, project="object-detection-results")
     
    print('\n\n#####################\n\nClassifying\n\n#####################')
    prologFacts = convert_yolo_output_to_partof_facts(get_path_to_newest_yolo_result("object-detection-results"))
    df = make_dataframe_from_prolog_facts(prologFacts) 
    df_class = classify(df, classifier)
    if df_class:
        print('Input is member of class')
    else:
        print('Input is not member of class')

    ################################################
    #### settings for calculation of near miss #### 
    ################################################
    # define search-space of near-miss algorithm (cells with these values will be subject to change in search of a near miss)
    combos = get_filter_combos(df) 

    # define permissible changes for each argument/column {'column-name of Data Frame': [list of values that selected cells will be changed to]}
    changesDict = {'arg1': ['blue', 'red', 'yellow'], 'arg2': ['triangle', 'circle', 'square'], 'arg3': ['triangle', 'circle', 'square', 'no_shape']}

    # only relevant for calculate_near_miss_hit: define weights of columns to be used in calculating the distance of near miss/hit to input
    column_weights = {'arg1': 1, 'arg2':1, 'arg3':1} # larger weights --> changes in respective argument/column produce greater distance
    
    ###################################
    #### near miss/hit calculation ####
    ###################################
    if df_class:
        print('\n\n#####################\n\nCalculating near miss\n\n#####################')
    else:
        print('\n\n#####################\n\nCalculating near hit\n\n#####################')

    # _, changesDF = calculate_near_miss_hit(df, combos, changesDict, column_weights, classifier) # get changes (as DataFrame) to input that make it a near miss/hit
    _, changesDF = calculate_near_miss_hit_2(df, combos, changesDict, classifier) # get changes (as DataFrame) to input that make it a near miss/hit

    df.columns = ['color', 'shape', 'part_of', 'x', 'y', 'w', 'h']
    changesDF.columns = ['color', 'shape', 'part_of', 'x', 'y', 'w', 'h']
    print(f'Object detection output of input image:\n{df}')
    print(f'\n\nChanges to input that result in the opposite classification:\n{changesDF}')

    ################################################################
    #### draw near miss/hit info in original image with changes ####
    ################################################################
    if df_class:
        print('\n\n#####################\n\nVisualizing near miss\n\n#####################')
    else:
        print('\n\n#####################\n\nVisualizing near hit\n\n#####################')

    changesList = changesDF.values.tolist() # convert DataFrame to List
    changesList = [concat_labels(elem) for elem in changesList] # format changesList so each elements in in form [label,x,y,w,h]
    img = cv2.imread(input)
    img = draw_bb(img, changesList)
    
    if df_class:
        cv2.imwrite('near-miss.png', img)
        print('Near miss visualization saved to near-miss.png')
    else:
        cv2.imwrite('near-hit.png', img)
        print('Near hit visualization saved to near-hit.png')


def parse_opt(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='path to yolo-model')
    parser.add_argument('--classifier',type=str, help='path to .pl-classifier')
    parser.add_argument('--input', type=str, help='path to input image')

    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)