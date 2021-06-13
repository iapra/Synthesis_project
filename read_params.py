## Read input data

import sys
import csv
import json

import import_open_datasets
import align_features
import roof_classification

def main():
	print("starting roof image classification")
	print("importing datasets")
	import_open_datasets.read_input("params.json")
	print("aligning features")
	align_features.main()
	print("image classification")
	roof_classification.main()
if __name__ == '__main__':
    main()
