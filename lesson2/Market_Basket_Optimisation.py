
import pandas as pd
import numpy as np

from  pandas  import DataFrame
import  xlrd


def rul():
    from efficient_apriori import apriori
    Market_Basket_Optimisation = pd.read_csv("Market_Basket_Optimisation.csv")
    transactions = []
    for i in range(Market_Basket_Optimisation.shape[0]):
        tem = []
        for v in Market_Basket_Optimisation.iloc[i]:
            if str(v) != "nan":
                tem.append(str(v))
        transactions.append(tem)

    itemsets,rules = apriori(transactions,min_support=0.02,min_confidence=0.02)
    print('频繁项集：', itemsets)
    print('关联规则：', rules)
rul()

