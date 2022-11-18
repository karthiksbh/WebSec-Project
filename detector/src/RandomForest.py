import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor

def main():
    #importing the dataset 
    dataset = pd.read_csv("detector/src/dataset/dataset.csv") 
    dataset = dataset.drop('index', 1)

    #removing unwanted column 
    x = dataset.iloc[ : , :-1].values 
    y = dataset.iloc[:, -1:].values

    #splitting the dataset into training set and test set 
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 42 )

    # ----------------applying grid search to find best performing parameters 
    parameters = [{'n_estimators': [100,700], 'max_features': ['sqrt', 'log2']}]
    # 'criterion' :['gini', 'entropy']}] 

    grid_search = GridSearchCV(RandomForestClassifier(), parameters,cv =5, n_jobs= -1)

    grid_search.fit(x_train, y_train.ravel()) 
    #printing best parameters
    print("Best Accuracy =" +str( grid_search.best_score_)) 
    print("best parameters =" + str(grid_search.best_params_))

    #fitting RandomForest regression with best params 
    classifier = RandomForestClassifier(n_estimators = 100, max_features = 'sqrt',random_state = 0) 
    classifier.fit(x_train, y_train.ravel())

    #predicting the tests set result 
    y_pred = classifier.predict(x_test)

    #confusion matrix 
    cm =confusion_matrix(y_test, y_pred) 
    print("Confusion Matrix: ")
    print(cm) 
    forest_regressor = RandomForestRegressor()
    forest_regressor.fit(x_train,y_train.ravel()) 
    prediction = forest_regressor.predict(x_test) 
    rmse = mean_squared_error(y_pred,y_test.ravel()) 
    print("Root Mean squared Error",rmse)

    # pickle file joblib 
    joblib.dump(classifier, 'final_model/rf_final.pkl')
    # #-------------Features Importance random forest 
    names = dataset.iloc[:,:-1].columns 
    importances =classifier.feature_importances_ 
    sorted_importances = sorted(importances, reverse=True)
    indices = np.argsort(-importances) 
    var_imp = pd.DataFrame(sorted_importances,names[indices], columns=['importance'])