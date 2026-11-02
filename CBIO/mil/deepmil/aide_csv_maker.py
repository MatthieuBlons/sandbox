import numpy as np
import pandas as pd
from functools import reduce
from sklearn.model_selection import StratifiedKFold


def random_sampling_in_df(df: pd.DataFrame, n: int | None = None, replace: bool =False):
    if n is None:
        n = len(df)
    indices = np.random.choice(df.index, n, replace=replace)
    return df.loc[indices]


def select_in_df(df: pd.DataFrame, feat: dict, n: int | None = None, replace: bool =False):
    bools = np.ones(len(df), dtype=bool)
    for col in feat:
        if isinstance(feat[col], list):
            tmp = []
            for v in feat[col]:
                tmp.append(np.logical_and(bools, df[col] == v))
            bools = reduce(np.logical_or, tmp)
        else:
            bools = np.logical_and(bools, df[col] == feat[col])
    selection = df[bools]
    return selection, len(selection)


def make_new_var(vars: str | list[str | None], separator: str | None = "_"):
    if isinstance(vars, str):
        return vars
    else:
        new = ""
        for v in vars:
            if v is not None:
                new += str(v) + separator
        return new



class DatasetMaker:
    def __init__(self, table, target, k=5):
        """
        equ_vars : nom des variables devant etre équilibré wrt target.
        fix_vars : dictio {variable : fix_value} pour fixer des variables.
        target : target for classification, used for balancing the dataset
        """
        self.table = pd.read_csv(table)
        self.target = target
        self.k = k

    def set_test(self, table, equ_vars):
        """
        divide into k-fold the dataset (creation of test_set)
        according to all the variables in equ_vars and target.
        """
        if type(equ_vars) != list:
            equ_vars = [equ_vars]
        equ_vars = equ_vars + [self.target]
        new_var = self.table.apply(make_new_var(equ_vars), axis=1)
        table["stratif"] = new_var
        skf = StratifiedKFold(n_splits=self.k, shuffle=True)
        y = table["stratif"].values
        X = list(range(len(y)))
        dic_test = dict()
        test_vec = np.zeros(len(y))
        for o, (train, test) in enumerate(skf.split(X, y)):
            for i in test:
                test_vec[i] = o
        table["test"] = test_vec
        return table

    def make_dataset(self, equ_vars: list, upsample=True, fix_vars=None):
        """make_dataset.

        returns a table with a stratified and balanced dataset.

        :param equ_vars: list of variable to balance amond the classes of the
        target variable.
        :type equ_vars: list
        :param upsample: strategy to sample the minority/majority class
        for balancing the dataset.
        :param fix_vars: dictionary setting the variables that you want to be fixed.
        """
        table = select_in_df(self.table, dictio=fix_vars)[0]
        table = self.set_test(table, equ_vars)
        new_var = self.table.apply(make_new_var(equ_vars), axis=1)
        table["equ"] = new_var
        lv_values = set(new_var.values)
        tmp = []
        for lv in lv_values:
            tmp_table = []
            tmp_N = []
            for tv in self.target_values:
                d = {"equ": lv, self.target: tv}
                sub_table = select_in_df(table, d)
                tmp_table.append(sub_table[0])
                tmp_N.append(sub_table[1])
            sup = max(tmp_N) if upsample else min(tmp_N)
            for tt in tmp_table:
                if len(tt) == sup:
                    tmp.append(tt)
                else:
                    tmp.append(random_sampling_in_df(tt, sup, replace=upsample))
        selection = pd.concat(tmp)
        return selection


def test_stratif(table, equ_vars, target, k):
    """
    divide into k-fold the dataset (creation of test_set)
    according to all the variables in equ_vars and target.
    """
    if isinstance(table, str):
        table = pd.read_csv(table)
    if type(equ_vars) != list:
        equ_vars = [equ_vars]
    equ_vars = equ_vars + [target]
    new_var = table.apply(make_new_var(equ_vars), axis=1)
    table["stratif"] = new_var
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    y = table["stratif"].values
    X = list(range(len(y)))
    dic_test = dict()
    test_vec = np.zeros(len(y))
    for o, (train, test) in enumerate(skf.split(X, y)):
        for i in test:
            test_vec[i] = o
    table["test"] = test_vec
    return table
