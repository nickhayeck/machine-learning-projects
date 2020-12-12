import numpy as np
import pandas as pd
np.random.seed(37)
import random
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder


# Dataset information
# the column names (names of the features) in the data files
col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
             'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
             'hours-per-week', 'native-country']
col_names_y = ['label']

numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                  'hours-per-week']
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'native-country']

# 1. Data loading from file and pre-processing.
def load_data(csv_file_path):
    raw = pd.read_csv(csv_file_path, names=np.concatenate((col_names_x,col_names_y)))

    y = raw.iloc[:,-1]
    enc1 = OrdinalEncoder()
    y = enc1.fit_transform(y.values.reshape(-1,1))

    categorical = raw[categorical_cols].replace([' Holand-Netherlands'],' ?')
    enc2 = OneHotEncoder(handle_unknown='ignore', sparse=False)
    categorical_onehot = enc2.fit_transform(categorical.values)
    numerical = raw[numerical_cols]
    numerical = (numerical - numerical.min())/(numerical.max()-numerical.min())

    x = np.concatenate((numerical.values,categorical_onehot), axis=1)
    # use if we need to get rid of null stuff obj_df[obj_df.isnull().any(axis=1)]

    return x, y

# 2. Select best hyperparameter with cross validation and train model.
def train_and_select_model(training_csv):
    # load data and preprocess from filename training_csv
    x_train, y_train = load_data(training_csv)
    # hard code hyperparameter configurations, an example:
    param_set = [
                 {'kernel': 'rbf', 'C': 1, 'degree': 1},
                 {'kernel': 'poly', 'C': 1, 'degree': 1},
                 {'kernel': 'linear', 'C': 1, 'degree': 1},
    ]
    # iterate over all hyperparameter configurations
    # perform 3 FOLD cross validation
    # print cv scores for every hyperparameter and include in pdf report
    # select best hyperparameter from cv scores, retrain model
    best_score = 0
    best_model = None

    for params in param_set:
        s,m = SVM(params,x_train,y_train)
        if s>best_score:
            best_score = s
            best_model = m

    return best_model, best_score

def SVM(params,x,y):
    model = SVC(C=params['C'], kernel=params['kernel'], degree=params['degree'], verbose=False)
    score = cross_val_score(model,x,y,cv=3)
    print(f"Score={np.mean(score):.2f} with params C={params['C']}, kernel={params['kernel']}, degree={params['degree']}. \nCV Testing Accuracies Follow:\n{score}")
    model.fit(x,y)

    return np.mean(score),model


# predict for data in filename test_csv using trained model
def predict(test_csv, trained_model):
    x_test, _ = load_data(test_csv)
    predictions = trained_model.predict(x_test)
    return predictions

# save predictions on test data in desired format
def output_results(predictions):
    with open('predictions.txt', 'w') as f:
        for pred in predictions:
            if pred == 0:
                f.write('<=50K\n')
            else:
                f.write('>50K\n')

if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    # return a trained model with best hyperparameter from 3-FOLD
    # cross validation to select hyperparameters as well as cross validation score for best hyperparameter.
    trained_model, cv_score = train_and_select_model(training_csv)

    # use trained SVC model to generate predictions
    predictions = predict(testing_csv, trained_model)
    output_results(predictions)
