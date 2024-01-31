
import argparse
import os
import glob
import pandas as pd
import numpy as np
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess data')
    parser.add_argument('--datadir', type=str, default='/trinity/home/mzijta/repositories/phd_project/EfficientAD_3D/mvtec_anomaly_detection/embryos_checked_split/test/defect/')
    parser.add_argument('--savedir', type=str, default='/trinity/home/mzijta/repositories/phd_project/EfficientAD_3D/mvtec_anomaly_detection/embryos_checked_split_week11/test/defect/')
    parser.add_argument('--attri_file', type=str, default='/trinity/home/mzijta/Nodig_prepro/ga_id_total.csv')
    parser.add_argument('--attributes', type=str, default=['GA'])
    parser.add_argument('--filter_week', type=int, default=11)
    args = parser.parse_args()
    return args


def main(args):
    files = sorted(os.listdir(args.datadir))
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
    data = pd.read_csv(args.attri_file)
    for file in files:
        predictnr = int(file[:5])
        week = int(file.split('US_')[1].split('_')[0])
        nr = file.split('_lca.nii')[0].split('_')[-1]
        print(predictnr, week, nr)

        df = data.loc[data['ID'] == predictnr]
        if len(df)>0:
          print(df)
          df = df.loc[data['weeknr'] == week]
          if len(df)>0:
              GA = int(df['GA'])
              GA_week = np.floor(GA/7)
              print(GA_week)
    
              if GA_week == args.filter_week:
                  shutil.copy(os.path.join(args.datadir,file),  os.path.join(args.savedir,file))
        else:
          if week == args.filter_week:
              shutil.copy(os.path.join(args.datadir,file),  os.path.join(args.savedir,file))
         
        
    
            


if __name__ == '__main__':
    args = parse_args()
    main(args)
   