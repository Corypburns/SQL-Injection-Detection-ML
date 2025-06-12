import subprocess

def naive_bayes():
    bayes_csic = '/home/cory/code/CISResearchSummer2025/CODE/CSIC-CODE-2010/CSIC-NaiveBayes.py'
    bayes_sqliv3 = '/home/cory/code/CISResearchSummer2025/CODE/SQLiV3-CODE/SQLiV3-NaiveBayes.py'
    bayes_gambleryu = '/home/cory/code/CISResearchSummer2025/CODE/GAMBLERYU-CODE/GAMBLERYU-NaiveBayes.py'
    bayes_sqli_extended = '/home/cory/code/CISResearchSummer2025/CODE/SQLi-EXTENDED-CODE/SQLiExtended-NaiveBayes.py'
    bayes_ghaemi = '/home/cory/code/CISResearchSummer2025/CODE/GHAEMI-CODE/GHAEMI-NaiveBayes.py'

    print("""
    1) Naive Bayes - CSIC
    2) Naive Bayes - SQLiV3
    3) Naive Bayes - GAMBLERYU
    4) Naive Bayes - SQLi-Extended
    5) Naive Bayes - GHAEMI
    """)

    input_ = int(input("-> "))

    match(input_):
        case 1:
            with open(bayes_csic) as f:
                exec(f.read())
        case 2:
            with open(bayes_sqliv3) as f:
                exec(f.read())
        case 3:
            with open(bayes_gambleryu) as f:
                exec(f.read())
        case 4:
            with open(bayes_sqli_extended) as f:
                exec(f.read())
        case 5:
            with open(bayes_ghaemi) as f:
                exec(f.read())
def xg_boost():
    xgboost_csic = '/home/cory/code/CISResearchSummer2025/CODE/CSIC-CODE-2010/CSIC-XGBoost.py'
    xgboost_sqliv3 = '/home/cory/code/CISResearchSummer2025/CODE/SQLiV3-CODE/SQLiV3-XGBoost.py'
    xgboost_gambleryu = '/home/cory/code/CISResearchSummer2025/CODE/GAMBLERYU-CODE/GAMBLERYU-XGBoost.py'
    xgboost_sqli_extended = '/home/cory/code/CISResearchSummer2025/CODE/SQLi-EXTENDED-CODE/SQLiExtended-XGBoost.py'
    xgboost_ghaemi = '/home/cory/code/CISResearchSummer2025/CODE/GHAEMI-CODE/GHAEMI-XGBoost.py' 

    print("""
    1) XGBoost - CSIC
    2) XGBoost - SQLiV3
    3) XGBoost - GAMBLERYU
    4) XGBoost - SQLi-Extended
    5) XGBoost - GHAEMI
    """)

    input_ = int(input("-> "))

    match(input_):
        case 1:
            with open(xgboost_csic) as f:
                exec(f.read())
        case 2:
            with open(xgboost_sqliv3) as f:
                exec(f.read())
        case 3:
            with open(xgboost_gambleryu) as f:
                exec(f.read())
        case 4:
            with open(xgboost_sqli_extended) as f:
                exec(f.read())
        case 5:
            with open(xgboost_ghaemi) as f:
                exec(f.read())

def ada_boost():
    ada_boost_csic = '/home/cory/code/CISResearchSummer2025/CODE/CSIC-CODE-2010/CSIC-AdaBoost.py'
    ada_boost_sqliv3 = '/home/cory/code/CISResearchSummer2025/CODE/SQLiV3-CODE/SQLiV3-AdaBoost.py'
    ada_boost_gambleryu = '/home/cory/code/CISResearchSummer2025/CODE/GAMBLERYU-CODE/GAMBLERYU-AdaBoost.py'
    ada_boost_sqli_extended = '/home/cory/code/CISResearchSummer2025/CODE/SQLi-EXTENDED-CODE/SQLiExtended-AdaBoost.py'
    ada_boost_ghaemi = '/home/cory/code/CISResearchSummer2025/CODE/GHAEMI-CODE/GHAEMI-AdaBoost.py'

    print("""
    1) Ada Boost - CSIC
    2) Ada Boost - SQLiV3
    3) Ada Boost - GAMBLERYU
    4) Ada Boost - SQLi-Extended
    5) Ada Boost - GHAEMI
    """)

    input_= int(input("-> "))

    match(input_):
        case 1:
            with open(ada_boost_csic) as f:
                exec(f.read())
        case 2:
            with open(ada_boost_sqliv3) as f:
                exec(f.read())
        case 3:
            with open(ada_boost_gambleryu) as f:
                exec(f.read())
        case 4:
            with open(ada_boost_sqli_extended) as f:
                exec(f.read())
        case 5:
            with open(ada_boost_ghaemi) as f:
                exec(f.read())

def g_boost():
    g_boost_csic = '/home/cory/code/CISResearchSummer2025/CODE/CSIC-CODE-2010/CSIC-GBoost.py'
    g_boost_sqliv3 = '/home/cory/code/CISResearchSummer2025/CODE/SQLiV3-CODE/SQLiV3-GBoost.py'
    g_boost_gambleryu = '/home/cory/code/CISResearchSummer2025/CODE/GAMBLERYU-CODE/GAMBLERYU-GBoost.py'
    g_boost_sqli_extended = '/home/cory/code/CISResearchSummer2025/CODE/SQLi-EXTENDED-CODE/SQLiExtended-GBoost.py'
    g_boost_ghaemi = '/home/cory/code/CISResearchSummer2025/CODE/GHAEMI-CODE/GHAEMI-GBoost.py'

    print("""
    1) GBoost - CSIC
    2) GBoost - SQLiV3
    3) GBoost - GAMBLERYU
    4) GBoost - SQLi-Extended
    5) GBoost - GHAEMI
    """)

    input_ = int(input("-> "))

    match(input_):
        case 1:
            with open(g_boost_csic) as f:
                exec(f.read())
        case 2:
            with open(g_boost_sqliv3) as f:
                exec(f.read())
        case 3:
            with open(g_boost_gambleryu) as f:
                exec(f.read())
        case 4:
            with open(g_boost_sqli_extended) as f:
                exec(f.read())
        case 5:
            with open(g_boost_ghaemi) as f:
                exec(f.read())

def random_forest():
    rforest_csic = '/home/cory/code/CISResearchSummer2025/CODE/CSIC-CODE-2010/CSIC-RandomForest.py'
    rforest_sqliv3 = '/home/cory/code/CISResearchSummer2025/CODE/SQLiV3-CODE/SQLiV3-RandomForest.py'
    rforest_gambleryu = '/home/cory/code/CISResearchSummer2025/CODE/GAMBLERYU-CODE/GAMBLERYU-RandomForest.py'
    rforest_sqli_extended = '/home/cory/code/CISResearchSummer2025/CODE/SQLi-EXTENDED-CODE/SQLiExtended-RandomForest.py'
    rforest_ghaemi = '/home/cory/code/CISResearchSummer2025/CODE/GHAEMI-CODE/GHAEMI-RandomForest.py'

    print("""
    1) Random Forest - CSIC
    2) Random Forest - SQLiV3
    3) Random Forest - GAMBLERYU
    4) Random Forest - SQLi-Extended
    5) Random Forest - GHAEMI
    """)

    input_ = int(input("-> "))

    match(input_):
        case 1:
            with open(rforest_csic) as f:
                exec(f.read())
        case 2:
            with open(rforest_sqliv3) as f:
                exec(f.read())
        case 3:
            with open(rforest_gambleryu) as f:
                exec(f.read())
        case 4:
            with open(rforest_sqli_extended) as f:
                exec(f.read())
        case 5:
            with open(rforest_ghaemi) as f:
                exec(f.read())

def sgd_classifier():
    sgd_csic = '/home/cory/code/CISResearchSummer2025/CODE/CSIC-CODE-2010/CSIC-SGDClassifier.py'
    sgd_sqliv3 = '/home/cory/code/CISResearchSummer2025/CODE/SQLiV3-CODE/SQLiV3-SGDClassifier.py'
    sgd_gambleryu = '/home/cory/code/CISResearchSummer2025/CODE/GAMBLERYU-CODE/GAMBLERYU-SGDClassifier.py'
    sgd_sqli_extended = '/home/cory/code/CISResearchSummer2025/CODE/SQLi-EXTENDED-CODE/SQLiExtended-SGDClassifier.py'
    sgd_ghaemi = '/home/cory/code/CISResearchSummer2025/CODE/GHAEMI-CODE/GHAEMI-SGDClassifier.py'

    print("""
    1) SGDClassifier - CSIC
    2) SGDClassifier - SQLiV3
    3) SGDClassifier - GAMBLERYU
    4) SGDClassifier - SQLi-Extended
    5) SGDClassifier - GHAEMI
    """)

    input_ = int(input("-> "))

    match(input_):
        case 1:
            with open(sgd_csic) as f:
                exec(f.read())
        case 2:
            with open(sgd_sqliv3) as f:
                exec(f.read())
        case 3:
            with open(sgd_gambleryu) as f:
                exec(f.read())
        case 4:
            with open(sgd_sqli_extended) as f:
                exec(f.read())
        case 5:
            with open(sgd_ghaemi) as f:
                exec(f.read())

def main():
    print("""
    1) Naive Bayes
    2) XGBoost
    3) AdaBoost
    4) GBoost
    5) Random Forest
    6) SGDClassifier
    """)

    input_ = int(input("-> "))

    match(input_):
        case 1:
            naive_bayes()
        case 2:
            xg_boost()
        case 3:
            ada_boost()
        case 4:
            g_boost()
        case 5:
            random_forest()
        case 6:
            sgd_classifier()

if __name__ == '__main__':
    main()
