import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


if __name__ == '__main__':
    df = pd.read_csv('input/train_targets_scored.csv')
    df.loc[:,"kfold"] = -1
    #performing  randomization 
    df = df.sample(frac=1).reset_index(drop=True)
    targets = df.drop("sig_id", axis=1).values

    #stratfied kfold for multilable classification

    mskf = MultilabelStratifiedKFold(n_splits=5)

    for fold,(trn, val) in enumerate(mskf.split(X=df, y=targets)):
        df.loc[val, "kfold"] = fold
    df.to_csv("train_targets_folds.csv", index=False)



