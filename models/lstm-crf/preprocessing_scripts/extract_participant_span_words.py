import os
import pprint
import itertools
from glob import glob

pp = pprint.PrettyPrinter()

def fname_to_pmid(fname):
	pmid = os.path.splitext(os.path.basename(fname))[0].split('.')[0]
	return pmid

def extract_participant_span_words():

	labels_to_preserve = ['1']
	
	tokens_fnames = glob('/Users/bhavnasaluja/Desktop/Spring2019/EBM-NLP_forked/EBM-NLP/ebm_nlp_2_00/documents/*.tokens')
	pos_fnames = glob('/Users/bhavnasaluja/Desktop/Spring2019/EBM-NLP_forked/EBM-NLP/ebm_nlp_2_00/documents/*.pos')

	train_fnames = glob('/Users/bhavnasaluja/Desktop/Spring2019/EBM-NLP_forked/EBM-NLP/ebm_nlp_2_00/annotations/aggregated/starting_spans/participants/train/*.ann')
	test_fnames = glob('/Users/bhavnasaluja/Desktop/Spring2019/EBM-NLP_forked/EBM-NLP/ebm_nlp_2_00/annotations/aggregated/starting_spans/participants/test/gold/*.ann')

	hierarchical_train_fnames = glob('/Users/bhavnasaluja/Desktop/Spring2019/EBM-NLP_forked/EBM-NLP/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels/participants/train/*.ann')
	hierarchical_test_fnames = glob('/Users/bhavnasaluja/Desktop/Spring2019/EBM-NLP_forked/EBM-NLP/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels/participants/test/gold/*.ann')

	pmid_to_list_of_indices_to_remove = {}

	for train_fname in train_fnames:
		print(train_fname)
		pmid = fname_to_pmid(train_fname)
		with open(train_fname, 'r') as f:
			train_ann_list = f.read().split()
			indices_to_remove = []
			for value in labels_to_preserve :
				for idx, train_ann in enumerate(train_ann_list):
					if train_ann != value:
						indices_to_remove.append(idx)

			pmid_to_list_of_indices_to_remove[pmid] = indices_to_remove

			train_ann_list = [v for i,v in enumerate(train_ann_list) if i not in indices_to_remove]

		with open(train_fname, 'w') as f:
			for i, word in enumerate(train_ann_list):
				if i != len(train_ann_list) - 1:
					f.write("{}\n".format(word))
				else:
					f.write(word)


	for test_fname in test_fnames:
		print(test_fname)
		pmid = fname_to_pmid(test_fname)
		with open(test_fname, 'r') as f:
			test_ann_list = f.read().split()
			indices_to_remove = []
			for value in labels_to_preserve :
				for idx, test_ann in enumerate(test_ann_list):
					if test_ann != value:
						indices_to_remove.append(idx)

			pmid_to_list_of_indices_to_remove[pmid] = indices_to_remove

			test_ann_list = [v for i,v in enumerate(test_ann_list) if i not in indices_to_remove]

		with open(test_fname, 'w') as f:
			for i, word in enumerate(test_ann_list):
				if i != len(test_ann_list) - 1:
					f.write("{}\n".format(word))
				else:
					f.write(word)


	for pos_fname in pos_fnames:
		print(pos_fname)
		pmid = fname_to_pmid(pos_fname)
		if pmid in pmid_to_list_of_indices_to_remove:
			indices_to_remove = pmid_to_list_of_indices_to_remove[pmid]
		else:
			continue
		with open(pos_fname, 'r') as f:
			pos_list = f.read().split()
			pos_list = [v for i,v in enumerate(pos_list) if i not in indices_to_remove]

		with open(pos_fname, 'w') as f:
			for i, word in enumerate(pos_list):
				if i != len(pos_list) - 1:
					f.write("{}\n".format(word))
				else:
					f.write(word)


	for token_fname in tokens_fnames:
		print(token_fname)
		pmid = fname_to_pmid(token_fname)
		if pmid in pmid_to_list_of_indices_to_remove:
			indices_to_remove = pmid_to_list_of_indices_to_remove[pmid]
		else:
			continue
		with open(token_fname, 'r') as f:
			token_list = f.read().split()
			token_list = [v for i,v in enumerate(token_list) if i not in indices_to_remove]

		with open(token_fname, 'w') as f:
			for i, word in enumerate(token_list):
				if i != len(token_list) - 1:
					f.write("{}\n".format(word))
				else:
					f.write(word)

	for token_fname in hierarchical_train_fnames:
		print(token_fname)
		pmid = fname_to_pmid(token_fname)
		if pmid in pmid_to_list_of_indices_to_remove:
			indices_to_remove = pmid_to_list_of_indices_to_remove[pmid]
		else:
			continue
		with open(token_fname, 'r') as f:
			token_list = f.read().split()
			token_list = [v for i,v in enumerate(token_list) if i not in indices_to_remove]

		with open(token_fname, 'w') as f:
			for i, word in enumerate(token_list):
				if i != len(token_list) - 1:
					f.write("{}\n".format(word))
				else:
					f.write(word)


	for token_fname in hierarchical_test_fnames:
		print(token_fname)
		pmid = fname_to_pmid(token_fname)
		if pmid in pmid_to_list_of_indices_to_remove:
			indices_to_remove = pmid_to_list_of_indices_to_remove[pmid]
		else:
			continue
		with open(token_fname, 'r') as f:
			token_list = f.read().split()
			token_list = [v for i,v in enumerate(token_list) if i not in indices_to_remove]

		with open(token_fname, 'w') as f:
			for i, word in enumerate(token_list):
				if i != len(token_list) - 1:
					f.write("{}\n".format(word))
				else:
					f.write(word)

if __name__ == '__main__':
	extract_participant_span_words()
