from yolov5 import detect
from utilities import convert_yolo_output_to_partof_facts, get_ground_truths_from_dir_structure
import argparse
import os
import shutil
from popper.util import Settings, print_prog_score, format_prog, order_prog
from popper.loop import learn_solution

def run(weights, source):
    if os.path.exists("yolo-train-results"):
        shutil.rmtree("yolo-train-results")

    # save yolo predictions
    print('\n\n#####################\n\nDetecting Objects\n\n#####################')
    detect.run(weights=weights, source=source + "/true", nosave=True, save_txt=True, project="yolo-train-results", name="t")
    detect.run(weights=weights, source=source + "/false", nosave=True, save_txt=True, project="yolo-train-results", name="f")
    detect.run(weights=weights, source=source + "/counterfactual", nosave=True, save_txt=True, project="yolo-train-results", name="cf")

    # generate popper files (bk.pl, bias.pl and exs.pl)
    os.makedirs("popper-files", exist_ok=True)
    bkFacts = convert_yolo_output_to_partof_facts("yolo-train-results/t/labels", "t") 
    bkFacts += convert_yolo_output_to_partof_facts("yolo-train-results/f/labels", "f")
    bkFacts += convert_yolo_output_to_partof_facts("yolo-train-results/cf/labels", "cf")

    with open('./popper-files/bk.pl', 'w') as bk: # create or overwrite file
        with open('bk-base.pl', 'r') as base:
            bk.write(base.read()) # copy bk-base.pl
        bk.write(bkFacts)

    with open('./popper-files/bias.pl', 'w') as bias: # create or overwrite file
        with open('bias-base.pl', 'r') as base: 
            bias.write(base.read()) # copy bias-base.pl

    exs = get_ground_truths_from_dir_structure(source) 
    with open('./popper-files/exs.pl', 'w') as exsFile: # create or overwrite file
        exsFile.write(exs)
        

    # learn and print rule with Popper
    print('\n\n#####################\n\nLearning Prolog Rule\n\n#####################')

    settings = Settings(kbpath='popper-files', debug=True)
    for i in range(10): # room for popper not finding a solution - popper doesn't seem to find a solution each time it runs
        prog, score, stats = learn_solution(settings)
        if prog != None:
            print_prog_score(prog, score)
            break
    if prog == None:
        print('NO SOLUTION')

    with open('learned-rule.pl', 'w') as rule: # create or overwrite file
        rule.write(format_prog(order_prog(prog)))
        print('\n\n#####################\n\nTheory written to learned-rule.pl\n\n#####################')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, help='model path(s)') 
    parser.add_argument('--source', type=str, help='directory of training images of form mainDir/(true or false or counterfactual)/image-name') 
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)