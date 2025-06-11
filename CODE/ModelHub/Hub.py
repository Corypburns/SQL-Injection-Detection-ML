import subprocess

def naive_bayes():
    bayes_csic = '/home/cory/code/CISResearchSummer2025/CODE/CSIC-CODE-2010/CSIC-NaiveBayes.py'
    bayes_sqliv3 = '/home/cory/code/CISResearchSummer2025/CODE/SQLiV3-CODE/SQLiV3-NaiveBayes.py'
    bayes_gambleryu = '/home/cory/code/CISResearchSummer2025/CODE/GAMBLERYU-CODE/GAMBLERYU-NaiveBayes.py'
    bayes_sqli_extended = '/home/cory/code/CISResearchSummer2025/CODE/SQLi-EXTENDED-CODE/SQLiExtended-NaiveBayes.py'


    print("""
    1) Naive Bayes - CSIC
    2) Naive Bayes - SQLiV3
    3) Naive Bayes - GAMBLERYU
    4) Naive Bayes - SQLi-Extended
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
def xg_boost():
    xgboost_csic = '/home/cory/code/CISResearchSummer2025/CODE/CSIC-CODE-2010/CSIC-XGBoost.py'
    xgboost_sqliv3 = '/home/cory/code/CISResearchSummer2025/CODE/SQLiV3-CODE/SQLiV3-XGBoost.py'
    xgboost_gambleryu = '/home/cory/code/CISResearchSummer2025/CODE/GAMBLERYU-CODE/GAMBLERYU-XGBoost.py'
    xgboost_sqli_extended = '/home/cory/code/CISResearchSummer2025/CODE/SQLi-EXTENDED-CODE/SQLiExtended-XGBoost.py'

    print("""
    1) XGBoost - CSIC
    2) XGBoost - SQLiV3
    3) XGBoost - GAMBLERYU
    4) XGBoost - SQLi-Extended
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

def ada_boost():
    ada_boost_csic = '/home/cory/code/CISResearchSummer2025/CODE/CSIC-CODE-2010/CSIC-AdaBoost.py'
    ada_boost_sqliv3 = '/home/cory/code/CISResearchSummer2025/CODE/SQLiV3-CODE/SQLiV3-AdaBoost.py'
    ada_boost_gambleryu = '/home/cory/code/CISResearchSummer2025/CODE/GAMBLERYU-CODE/GAMBLERYU-AdaBoost.py'
    ada_boost_sqli_extended = '/home/cory/code/CISResearchSummer2025/CODE/SQLi-EXTENDED-CODE/SQLiExtended-AdaBoost.py'

    print("""
    1) Ada Boost - CSIC
    2) Ada Boost - SQLiV3
    3) Ada Boost - GAMBLERYU
    4) Ada Boost - SQLi-Extended
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

def main():
    print("""
    1) Naive Bayes
    2) XGBoost
    3) AdaBoost
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
            with open(xgboost_csic) as f:
                exec(f.read())
        case 5:
            with open(xgboost_sqliv3) as f:
                exec(f.read())
        case 6:
            with open(xgboost_gambleryu) as f:
                exec(f.read())

if __name__ == '__main__':
    main()
