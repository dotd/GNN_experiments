from os import path
import pandas as pd


def prepare_csv(func):
    def inner(args):
        csv_file = args.csv_file
        if not path.exists(csv_file):

            df = pd.DataFrame(columns=[
                'pruning method',
                'architecture',
                'keep edges',
                'train acc',
                'test acc',
                'train time',
                'test time',
                ])
            df.to_csv(csv_file, index=False)
        func(args)


    return inner
