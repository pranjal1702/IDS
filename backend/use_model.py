from IDS import IntrusionDetector
from argparse import ArgumentParser
import pandas as pd
import numpy as np 

def use_model(args):
    intrusion = IntrusionDetector(model_path=args.modelpath, n_estimators=15)
    df = pd.read_csv(args.datapath)
    x, y = df.values[:, :-1], df.values[:, -1]
    y = np.array([list(intrusion.classes.values()).index(c) for c in y]).reshape(-1, 1)
    pred = intrusion.predict(x)
    intrusion.plot_metrics(y, pred)
    print(intrusion.predict(x, get_class_names=True))
    
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--datapath", "-d", help="Dataset File Path")
    parser.add_argument("--modelpath", "-m", default="intrusion-detector.pkl", help="Model File Path")
    args = parser.parse_args()
    use_model(args)
