#from models import LightBertBasedClassifier as lbbc
#from models import *
import argparse


ap = argparse.ArgumentParser(
    description='This script is for train LBBC Network using a specific dataset',
    prog='Light Bert-Based Classifier v0.1',
    epilog='See https://github.com/valgi0/LightBertBasedClassifier',
)

ap.add_argument(
    '--batch',
    default=32,
    action='store',
    type=int,
    help='Set the batch size for training',
)

ap.add_argument(
    '--epoch',
    default=30,
    action='store',
    type=int,
    help='Number of epochs for training',
)

ap.add_argument(
    '--save_point',
    default=10,
    action='store',
    type=int,
    help='Specify when save',
)

ap.add_argument(
    '--lr',
    default=6e-5,
    action='store',
    type=float,
    help='Lerning rate'
)

ap.add_argument(
    '--config',
    default='./config.json',
    action='store',
    type=str,
    help='Configuration file for the Net'
)

args = ap.parse_args()




                    
  
