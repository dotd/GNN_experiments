from os import path
import pandas as pd

csv_file = 'synthetic_results.csv'


def prepare_csv(func):
    def inner(args):
        if not path.exists(csv_file):

            df = pd.DataFrame(columns=[
                'keep edges',
                'minhash train',
                'minhash test',
                'minhash time train',
                'minhash time test',
                'random train',
                'random test',
                'random time train',
                'random time test',
                ])
            df.to_csv(csv_file, index=False)
        func(args)

    return inner
