# Concept Learner and Near-Miss/Hit Generator for composite Kandinsky-Patterns

Learn a classification-rule for complex Kandinsky Patterns involving large objects that are composed of small objects. Classify Kandinsky Figures accoring to that rule. Explain classifications and scrutinize the training dataset by visualizing near misses/hits. 

[yolov5](https://github.com/ultralytics/yolov5) is used for object detection.
[Popper](https://github.com/logic-and-learning-lab/Popper) is used as an ILP (inductive logic programming) paradigm to learn a classification rule. 

Yolo/Popper can in principle be replaced by other object-dection/ILP methods.


## Dataset

A [Kandinsky Figure](https://www.sciencedirect.com/science/article/pii/S0004370221000977) is an image consisting of any number of non-overlapping geometric shapes that are defined by their shape, size, color and location within the image. A Kandinsky Pattern is a concept within the world of Kandinsky Figures. As any other concept, it can be defined extensionally by a subset of all possible Kandinsky Figures, or intensionally, by defining the necessary and sufficient conditions for this concept. 

The Kandinsky Pattern used as an example in this implementation is subject of a challenge described in the above paper and the dataset can be found [here](https://github.com/human-centered-ai-lab/dat-kandinsky-patterns/tree/master/challenge-nr-1). This is the ground truth given by the authros of the challenge: "In a Kandinsky Figure small objects are arranged on big shapes that are the same as object shapes, in the big shape of type X, no small object of type X exists. Big square shapes only contain blue and red objects, big triangle shapes only contain yellow and red objects and big circle shapes contain only yellow and blue objects"


## Requirements

- [Popper](https://github.com/logic-and-learning-lab/Popper) and its dependencies; if not already present, clone the Popper repository in this repository

- [yolov5](https://github.com/ultralytics/yolov5) and its dependencies; if not already present, clone the yolov5 repository in this repository

- [OpenCV](https://opencv.org/)

- [pandas](https://pandas.pydata.org/)

A conda environment that worked for me can be found in conda-env.yaml. Install with this command: `conda env create -f ./conda-env.yaml`


## Folders and Files

- examples: contains an example-image of each class

- sample-data: a random sample of 10 images per class from the [composite-Kandinsky-Patterns-dataset](https://github.com/human-centered-ai-lab/dat-kandinsky-patterns/tree/master/challenge-nr-1)

- popper-files (created by learn-rule.py): include bias.pl (information to restrict the search space of Popper), bk.pl (background knowledge consisting of object-detection output and the definition of predicates relevant to the problem domain), exs.pl (examples based on the training-data)

- yolo-train-results (created by learn-rule.py): yolo output of training data

- object-detection-results (created by classify-and-near-miss-hit.py): yolo-output of the instance to be classified

- bias-base.pl: the search space restriction used to learn a classification rule with Popper

- bk-base.pl: prolog predicates relevant to the problem domain

- learned-rule.pl (created by learn-rule.py): the classifcation rule

- learn-rule.py: script to learn a classification rule from visual examples

- classify-and-near-miss-hit.py: script to classify an intance and produce a near-miss/hit 

- tests.py: unit-tests

- utilities.py: various helper functions used in learn-rule.py and classify-and-near-miss-hit.py

- kp-yolo-model.pt: a yolo model to detect objects in Kandinsky Figures (trained on 15000 randomly generated Kandinsky Figures with around 60 objects each) 

- conda-env.yaml: can be used to create anaconda environment (see Requirements)


## Usage Example

You can train a classifier like this: 
`python learn-rule.py --weights kp-yolo-model.pt --source sample-data`

Youn can classify an image like this: 
`python classify-and-near-miss-hit.py --weights kp-yolo-model.pt --classifier learned-rule.pl --input examples/example-cf.png`


## Notes on calculating a near miss/hit

There are functions to calculate a near miss/hit in utilities.py:

- calculate_near_miss_hit: very slow, but more exhaustive search and supports relative weighting of aspects in the input image

- calculate_near_miss_hit_2: faster, but does not support weighting

To switch between the two functions, uncomment one and comment out the other function in classify-and-near-miss-hit.py. 

The weights can also be adjusted in this file. To change the weights, change the values of the dictionary with name column_weights. The keys correspond to the following aspects of an image:

- arg1: the color of a small object

- arg2: the shape of a small object

- arg3: the shape of the large object that the small object is part of

