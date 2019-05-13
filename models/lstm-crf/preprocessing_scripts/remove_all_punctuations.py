import os
import pprint
import itertools
from glob import glob
import string

pp = pprint.PrettyPrinter()

def fname_to_pmid(fname):
	pmid = os.path.splitext(os.path.basename(fname))[0].split('.')[0]
	return pmid

def remove_punctuations_from_dataset():

	# punctuations = ['.']
	punctuations = string.punctuation
	print("Removing punctuations from the dataset = ", punctuations)
	
	tokens_fnames = glob('/Users/bhavnasaluja/Desktop/Spring2019/EBM-NLP_forked/EBM-NLP/ebm_nlp_2_00/documents/*.tokens')
	pos_fnames = glob('/Users/bhavnasaluja/Desktop/Spring2019/EBM-NLP_forked/EBM-NLP/ebm_nlp_2_00/documents/*.pos')

	train_fnames = glob('/Users/bhavnasaluja/Desktop/Spring2019/EBM-NLP_forked/EBM-NLP/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels/participants/train/*.ann')
	test_fnames = glob('/Users/bhavnasaluja/Desktop/Spring2019/EBM-NLP_forked/EBM-NLP/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels/participants/test/gold/*.ann')

	pmid_to_punctuation_indices = {}

	for tokens_fname in tokens_fnames:
		print(tokens_fname)
		pmid = fname_to_pmid(tokens_fname)
		with open(tokens_fname, 'r') as f:
			tokens_list = f.read().split()
			indices_to_remove = []
			for value in punctuations :
				for idx, token in enumerate(tokens_list):
					if token == value:
						indices_to_remove.append(idx)
			  # tokens_list = list(filter(lambda a: a != value, tokens_list))
			pmid_to_punctuation_indices[pmid] = indices_to_remove

			tokens_list = [v for i,v in enumerate(tokens_list) if i not in indices_to_remove]

		with open(tokens_fname, 'w') as fout:
			for token in tokens_list:
				fout.write('{}'.format(token))
				fout.write('\n')


	for pos_fname in pos_fnames:
		print(pos_fname)
		pmid = fname_to_pmid(pos_fname)
		if pmid in pmid_to_punctuation_indices:
			indices_to_remove = pmid_to_punctuation_indices[pmid]
		else:
			continue
		with open(pos_fname, 'r') as f:
			pos_list = f.read().split()
			pos_list = [v for i,v in enumerate(pos_list) if i not in indices_to_remove]

		with open(pos_fname, 'w') as fout:
			for pos in pos_list:
				fout.write('{}'.format(pos))
				fout.write('\n')


	for train_fname in train_fnames:
		print(train_fname)
		pmid = fname_to_pmid(train_fname)
		if pmid in pmid_to_punctuation_indices:
			indices_to_remove = pmid_to_punctuation_indices[pmid]
		else:
			continue
		with open(train_fname, 'r') as f:
			train_ann_list = f.read().split()
			train_ann_list = [v for i,v in enumerate(train_ann_list) if i not in indices_to_remove]

		with open(train_fname, 'w') as fout:
			for train_ann in train_ann_list:
				fout.write('{}'.format(train_ann))
				fout.write('\n')


	for test_fname in test_fnames:
		print(test_fname)
		pmid = fname_to_pmid(test_fname)
		if pmid in pmid_to_punctuation_indices:
			indices_to_remove = pmid_to_punctuation_indices[pmid]
		else:
			continue
		with open(test_fname, 'r') as f:
			test_ann_list = f.read().split()
			test_ann_list = [v for i,v in enumerate(test_ann_list) if i not in indices_to_remove]

		with open(test_fname, 'w') as fout:
			for i, test_ann in enumerate(test_ann_list):
				if i != len(test_ann_list) - 1:
					fout.write('{}\n'.format(test_ann))
				else:
					fout.write('{}'.format(test_ann))

if __name__ == '__main__':
	remove_punctuations_from_dataset()
