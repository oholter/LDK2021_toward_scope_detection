from src.io_handler import read_requirements
from argparse import ArgumentParser
import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--req',
                        help="requirements file",
                        action='store',
                        default="../data/raw_text/DNVGL-RU-FD.tsv")
    parser.add_argument('--save',
                        help="file to save samples",
                        default="DNVGL-OS-E101_samples.tsv")
    parser.add_argument('--num',
                        help='number of samples',
                        action='store',
                        default='200')
    args = parser.parse_args()


    df = read_requirements(args.req)

    try:
        samples = df.sample(n=int(args.num))
        print(samples)
        samples.to_csv(args.save, sep='\t', header=True, index=False, encoding='utf8')

    except Exception:
        print("num must be an integer")