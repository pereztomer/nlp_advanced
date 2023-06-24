from sklearn.preprocessing import LabelEncoder
import pandas as pd
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from xgboost import XGBClassifier


def evaluate_pred(pred, y_val):
    accuracy = metrics.accuracy_score(y_val, pred)
    precision = metrics.precision_score(y_val, pred, average='macro')
    recall = metrics.recall_score(y_val, pred, average='macro')
    F1 = 2 * ((precision * recall) / (precision + recall))
    print("acuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("F1: ", F1)


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod(
            (datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' %
              (thour, tmin, round(tsec, 2)))




def drop_columns_by_number(arr, columns_to_drop):
    # Drop the specified columns from the NumPy array
    arr = np.delete(arr, columns_to_drop, axis=1)


def tree_fitter(x_train, y_train, x_val, y_val):
    print("Xgboost train...")
    params = {
        'min_child_weight': [0.5, 1, 5, 10],
        'gamma': [0.1, 0.5, 1, 1.5, 2, 5],
        'subsample': [0.4, 0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.2],
        'reg_lambda': [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 200],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5,6]
    }
    # params = {
    #     'min_child_weight': [1, 5],
    #     'gamma': [2],
    #     'subsample': [0.6, 0.8],
    #     'reg_alpha': [0.1],
    #     'reg_lambda': [0, 0.1],
    #     'colsample_bytree': [0.6],
    #     'max_depth': [5, 6]
    # }

    xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='multi:softmax',
                        nthread=1)
    folds = 3
    param_comb = 5

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

    random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=16, scoring='accuracy', n_jobs=4,
                                       cv=skf.split(x_train, y_train), random_state=1001)

    start_time = timer(None)

    random_search.fit(x_train, y_train)
    timer(start_time)

    results = pd.DataFrame(random_search.cv_results_)
    results.to_csv('xgb-random-grid-search-results-01.csv', index=False)
    print(random_search.best_params_)

    predicted_y = random_search.best_estimator_.predict(x_val)

    print("XGBOOST :")
    evaluate_pred(pred=predicted_y, y_val=y_val)
    print("...................")
    # get_feature_importance_order(random_search.best_estimator_)
    return predicted_y



def get_feature_importance_order(clf):
    # Get the feature importance from the XGBoost classifier
    importance = clf.feature_importances_

    # Create a list of tuples with feature names and importance scores
    feature_importance = [(f'f{i}', importance_score) for i, importance_score in enumerate(importance)]

    # Sort the feature importance scores in descending order
    feature_importance_sorted = sorted(feature_importance, key=lambda x: x[1], reverse=True)

    # Get a list of feature names in order of their importance scores
    feature_names_ordered = [feature[0] for feature in feature_importance_sorted]
    print(feature_names_ordered)



def show_Statistics(df):
    st = df.describe()
    print(st.to_string())
    hist = df.hist(bins=100, ylabelsize=10, xlabelsize=10, figsize=(30, 18))
    # plt.savefig('my_plot.png')
    print("stats")


def create_ds(csv_file="data_team4_embeddings.csv.csv"):
    # Load CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Extract 'glove_embeddings' column as X features
    x = df['glove_embeddings'].values

    # Convert 'glove_embeddings' strings to numpy arrays
    df['glove_embeddings'] = df['glove_embeddings'].apply(lambda x: np.array(x[1:-1].split(), dtype=np.float32))

    # Extract 'glove_embeddings' column as X features
    x = np.stack(df['glove_embeddings'].values)*10
    # Extract 'label' column as y labels
    y = df['label'].values

    le = LabelEncoder()
    y = le.fit_transform(y)
    # Perform random horizontal split into train and validation sets
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, y_train, x_validation, y_validation





def Random_Forest(x_train, y_train, x_val, y_val):
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    pred = rf.predict(x_val)
    print("Random_Forest :")
    evaluate_pred(pred=pred, y_val=y_val)
    print("...................")
    return pred


if __name__ == '__main__':
    x_train, y_train, x_validation, y_validation = create_ds("data_team4_embeddings.csv.csv")

    # Random_Forest(x_train=X_train,y_train=Y_train, x_val=X_Test,y_val=Y_test)
    tree_fitter(x_train=x_train, y_train=y_train, x_val=x_validation, y_val=y_validation)