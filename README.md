# SQLi Detection With Machine Learning





<br><br>
## Step 1: Prerequesite Installation - What You Need
* Python
* Pip
* Python Packages
<br><br><br>

First, you need to install Python. You can do this by typing the following:


```
sudo apt install python
```
<br>

> ![info-image](https://github.com/Corypburns/SQL-Injection-Detection-ML/blob/master/Info.svg) **__IMPORTANT__**:  Installing Python also installs Pip. We took care of two birds with one stone with one step.
<br><br>

<br><br>Now, you need to use Pip and install the packages that are included inside of the "libraries.txt" file located within the repository. To install these packages, simply type:


```
pip install libraries.txt
```
<br><br><br>




## Step 2: Using the Models
In this part, I will show you how to use the models that are included within this repository. Documentation will be provided in the form of a link, which will lead to more information about the specific model. You can click on them from the bulleted list. The models used are as follows:
* [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
* [XGBoost](https://xgboost.readthedocs.io/en/release_3.2.0/)
* [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
* [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [GBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
<br><br>

> ![info-image](https://github.com/Corypburns/SQL-Injection-Detection-ML/blob/master/Info.svg) **__IMPORTANT__**:  I will be using an example to get you started on how to set the correct paths for your personal machine. The codebase I will be providing information from the XGBoost codebase.
<br><br>

<br><br>You will notice two things located towards the top of the file under the packages. In my case, they are as follows:


```Python
output_path = '/home/cory/code/CISResearchSummer2025/Outputs/XDGBoost/XGBoost-SQLiExtended'
dataset_path = '/home/cory/code/CISResearchSummer2025/DATASETS/SQLi-Extended/sqli-extended.csv'
```
<br><br><br>




In your case, they should look like this for linux users:


```Python
output_path = '/your/output/location'
dataset_path = '/your/dataset/location'
```
<br><br>

and for windows users:


```Python
output_path = r'DriveLetter:\your\output\location/'
dataset_path = r'DriveLetter:\your\dataset\location/file.csv'
```
<br><br><br>




Once your paths are in order, I will show you what lines are important to a model and its effort to classify the data that is provided from the dataset in question. First, you'll see variables that look like this:


```Python
df = pd.read_csv(dataset_path) # This loads the dataset using Pandas.

df = df[['Sentence', 'Label']].dropna() # This takes into account the data that you are feeding the model and the already-defined label provided in the dataset. Additionally, this will drop values that are incomplete or irrelevant; this boosts model performance.

df['Sentence'] # This is the data being fed, as stated above.

df['Label'] # This is the label of that data located towards the end of the row.

input_data = df['Sentence'] # Assigning the column called 'Sentence' to a variable.

label_data = df['Label'] # Assigning the column called 'Label' to a variable.
```
<br><br>

> ![info-image](https://github.com/Corypburns/SQL-Injection-Detection-ML/blob/master/Info.svg) **__IMPORTANT__**:  It is dire that the the "Label" column remain unknown to the model, otherwise the model will perform perfectly, making the process of testing its accuracy useless.
<br><br>


<br><br>Next, you will see this line of code regarding XGBoost (Note that each model has its corresponding method that is used):


```Python
X_train, X_test, y_train, y_test = train_test_split(input_data, label_data, test_size=0.2, random_state=42) # This sets the four variables assigned on the left to the variables we assigned in the earlier step and ties them together.
```
<br><br>

What this is going to do is split the dataset 80-20, where 80% is for training, and 20% is for testing. It is best practice to make the larger percentage of the dataset for training purposes. The split is the same across all models, but it can be adjusted to whichever split you would like; just make sure there is enough data for the training (having a larger testing value compared to the training values will yield terrible results.)
