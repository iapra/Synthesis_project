## Read input data

import sys
import csv
import json

import import_open_datasets
import align_features
import roof_classification

def main():
    import_open_datasets.read_input("params2.json")
    align_features.main()
    roof_classification.main()
if __name__ == '__main__':
    main()
