import os
import pprint
import itertools
from glob import glob

pp = pprint.PrettyPrinter()

def preserve_only_condition_labels():

    non_condition_labels = ['1', '2', '3']

    train_ann_fnames = glob('/Users/bhavnasaluja/Desktop/Spring2019/EBM-NLP_forked/EBM-NLP/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels/participants/train/*.ann')
    test_ann_fnames = glob('/Users/bhavnasaluja/Desktop/Spring2019/EBM-NLP_forked/EBM-NLP/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels/participants/test/gold/*.ann')
    
    for fname in train_ann_fnames:
        print("Reading file: ", fname)
        
        # words = list()

        with open(fname, 'r') as f:
            text = f.read().split()
            for value in non_condition_labels :
                text = [w.replace(value, '0') for w in text]
            # print("Text: ", text)


        with open(fname, 'w') as f:
            for i, label in enumerate(text):
                if i != len(text) - 1:
                    f.write("{}\n".format(label))
                else:
                    f.write(label)


    for fname in test_ann_fnames:
        print("Reading file: ", fname)
        
        # words = list()

        with open(fname, 'r') as f:
            text = f.read().split()
            for value in non_condition_labels :
                text = [w.replace(value, '0') for w in text]
            # print("Text: ", text)


        with open(fname, 'w') as f:
            for i, label in enumerate(text):
                if i != len(text) - 1:
                    f.write("{}\n".format(label))
                else:
                    f.write(label)

if __name__ == '__main__':
    preserve_only_condition_labels()
