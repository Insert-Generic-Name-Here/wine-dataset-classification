## Importing Essential Libraries and Modules
import os
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

## Defining Essential Functions
def quality_labels(y):
    if y <= 4:
        return -1
    elif y <= 6:
        return 0
    else:
        return 1

def association_rule_mining_red_wine(rn_wine):
    rn_wine.quality = np.array(list(map(quality_labels, rn_wine.quality)))

    ### Preprocessing
    ### * Converting Numerical Data to Categorical Data through 
    ###   Binning into 7 buckets for each feature
    for feature in rn_wine.iteritems():
        feature_name = feature[0]
        feature_values = feature[1]
        feature_type = feature[1].values.dtype
        
        if feature_type == 'float64':
            rn_wine[feature_name] = pd.cut(rn_wine[feature_name], 7)

    ## Executing the Apriori Algorithm
    ## * Trying to do something with the data
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules

    rn_wine_onehot = []

    for feature in rn_wine.iteritems():
        feature_name = feature[0]
        rn_wine_onehot.append(pd.get_dummies(rn_wine[feature_name], prefix=feature_name, prefix_sep='_'))

    for i in range(len(rn_wine_onehot)):
        try:
            rn_wine_onehot[i].columns = [str(j) for j in rn_wine_onehot[i].columns.categories]
        except AttributeError:
            rn_wine_onehot[i].columns = [str(j) for j in rn_wine_onehot[i].columns]

    rn_wine_onehot = pd.concat(rn_wine_onehot, axis=1)

    frequent_itemsets = apriori(rn_wine_onehot, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0)

    return rules


if __name__ == '__main__':
    ### Defining Cache Directories
    np_cache_dir = os.path.join('..', 'numpy_cache')
    csv_cache_dir = os.path.join('..', 'csv_cache')

    ### Reading - and Presenting - the Data
    ### * Vinho Verde Red Wine; Normalized; Outlier-Free
    rn_wine = pd.read_csv(os.path.join(csv_cache_dir, 'red_clean.csv'), sep='\t').drop(['Unnamed: 0'], axis=1)
    association_rule_mining_red_wine(rn_wine) 