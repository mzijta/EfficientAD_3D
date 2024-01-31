import argparse
import os
import glob
import pandas as pd
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess data')
    parser.add_argument('--datadir', type=str, default='/trinity/home/mzijta/repositories/phd_project/EfficientAD_3D/mvtec_anomaly_detection/embryos_checked_split/')
    args = parser.parse_args()
    return args

def main(args):
    datadir = args.datadir
    train_dir = os.path.join(datadir, 'train/good')
    test_dir = os.path.join(datadir, 'test/good')  
    val_dir = os.path.join(datadir, 'val/good')
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)


    csv_file = pd.read_csv(datadir + 'datasplit_.csv')
    csv_file_train = csv_file['train'].dropna()
    csv_file_train = [i for i in csv_file_train if i.endswith('.gz')]
    csv_file_test = csv_file['test'].dropna()
    csv_file_test = [i for i in csv_file_test if i.endswith('.gz')]
    csv_file_val = csv_file['val'].dropna()
    csv_file_val = [i for i in csv_file_val if i.endswith('.gz')]
    for filename in csv_file_train:
        print(filename)
        shutil.copy(filename, train_dir)
    for filename in csv_file_test:
        print(filename)
        shutil.copy(filename, test_dir)
    for filename in csv_file_val:
        print(filename)
        shutil.copy(filename, val_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)