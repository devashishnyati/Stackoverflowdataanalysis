import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import svm

from scipy.sparse import csr_matrix

df = pd.read_csv('/data/survey_results_public.csv', low_memory=False ,index_col='Respondent')
print(df.head())


def reduce_columns_salary(df):
    salary_columns = ['ConvertedSalary','Hobby','OpenSource','Country','Employment','FormalEducation','UndergradMajor','CompanySize','DevType','YearsCoding','YearsCodingProf','DatabaseWorkedWith','PlatformWorkedWith','FrameworkWorkedWith','OperatingSystem','Gender','Age']
    df_salary = df[salary_columns]
    df_salary_nans = df_salary[(df_salary.ConvertedSalary > 1000) & (df_salary.ConvertedSalary <= 250000)]
    return df_salary_nans

def other_categories(category, cutoff):
    category_map = {}
    for i in range(len(category)):
        if category.values[i] >= cutoff:
            category_map[category.index[i]] = category.index[i]
        else:
            category_map[category.index[i]] = 'Other'
    return category_map



"""
def split_data(df, column_name):
    s = df[column_name]
    t = s.str.split(';')
    df_split = pd.get_dummies(t.apply(pd.Series).stack()).sum(level=0)
    df.drop(column_name, inplace=True, axis=1)
    df = pd.merge(df, df_split, on='Respondent')
    return df
"""

df_salary_nans = reduce_columns_salary(df)
country_map = other_categories(df_salary_nans.Country.value_counts(), 400)
df_salary_nans['Shorted_Countries'] = df_salary_nans.Country.map(country_map)
lookup_education = {"Bachelor’s degree (BA, BS, B.Eng., etc.)": "Bachelors","Some college/university study without earning a degree": "Some college","Master’s degree (MA, MS, M.Eng., MBA, etc.)": "Masters","Associate degree": "Associate Degree","Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "High School","Professional degree (JD, MD, etc.)": "Professional","Other doctoral degree (Ph.D, Ed.D., etc.)": "Doctoral","nan": "nan","Primary/elementary school": "Elementary","I never completed any formal education": "No Education"}
df_salary_nans['Education'] = df_salary_nans.FormalEducation.map(lookup_education)


df2_salary = df_salary_nans.copy()
del(df2_salary['Country'])
del(df2_salary['FormalEducation'])
df2_salary.rename({'Shorted_Countries':'Country','Education':'FormalEducation'}, inplace=True)

labels_salary = df2_salary['ConvertedSalary']
features_salary = df2_salary.drop('ConvertedSalary', axis=1)

dummies_salary = pd.get_dummies(features_salary)
dum_columns_salary = dummies_salary.columns

train_features_salary, test_features_salary, train_labels_salary, test_labels_salary = train_test_split(dummies_salary, labels_salary, test_size=0.2, random_state=42)

"""
#SVR
param_grid = [{'kernel':('linear', 'rbf'), 'C':[1, 10]}]
regressor = SVR()
gridsearch = GridSearchCV(regressor, param_grid, scoring='neg_mean_squared_error')

regressor = gridsearch.best_estimator_

train_predictions = regressor.predict(train_regressor_ready_data)

rootMeanSquaredError_train = np.sqrt(mean_squared_error(train_labels_salary, train_predictions))
print("${:,.02f}".format(rootMeanSquaredError_train))

test_regressor_ready_data = csr_matrix(test_features.values)
test_predictions = regressor.predict(test_regressor_ready_data)

rootMeanSquaredError_test = np.sqrt(mean_squared_error(test_labels_salary, test_predictions))
print("${:,.02f}".format(rootMeanSquaredError_test))
"""

#Dimentionality Reduction
print('Dimentionality Reduction for LR')
pca_salary = PCA(n_components=100)
principalComponents = pca_salary.fit_transform(dummies_salary)
train_pca_salary = pca_salary.transform(train_features_salary)
test_pca_salary = pca_salary.transform(test_features_salary)

#Linear Regression
print('LR')
lr = LinearRegression()
reg_lr = lr.fit(train_pca_salary, train_labels_salary)
predict_salary = lr.predict(test_pca_salary)

rootMeanSquaredError_predict_salary = np.sqrt(mean_squared_error(test_labels_salary, predict_salary))
print("${:,.02f}".format(rootMeanSquaredError_predict_salary))




"""
#For new input
pred = features.iloc[4,:]
pred_df = pred.to_frame().T
pred_dummy_salary = pd.get_dummies(pred_df)
feature_difference = set(dummies) - set(pred_dummy)
feature_difference_df = pd.DataFrame(data=np.zeros((pred_dummy.shape[0], len(feature_difference))),columns=list(feature_difference))
pred_dummy_test = pred_dummy.join(feature_difference_df)
pred_dummy_test_fill = pred_dummy_test.fillna(0)
pred_dummy_test_fill = pred_dummy_test_fill[dum_columns]
test_regressor_pred_data = csr_matrix(pred_dummy_test_fill.values)
test_predictions_dummy = regressor.predict(test_regressor_pred_data)
print(test_predictions_dummy)
print(labels[11])

"""


def reduce_columns_job(df):
    satisfaction_columns = ['JobSatisfaction','ConvertedSalary','Hobby','OpenSource','Country','Employment','FormalEducation','UndergradMajor','CompanySize','DevType','YearsCoding','YearsCodingProf','DatabaseWorkedWith','PlatformWorkedWith','FrameworkWorkedWith','OperatingSystem','Gender','Age']
    df_satisfaction = df[satisfaction_columns]
    df_satisfaction_nnas = df_satisfaction.dropna(subset=['JobSatisfaction'])   
    return df_satisfaction_nnas


df_satisfaction_nnas = reduce_columns_job(df)

df_satisfaction_nnas['Shorted_Countries'] = df_satisfaction_nnas.Country.map(country_map)
df_satisfaction_nnas['Education'] = df_satisfaction_nnas.FormalEducation.map(lookup_education)

df2_job = df_satisfaction_nnas.copy()
del(df2_job['Country'])
del(df2_job['FormalEducation'])
df2_job.rename({'Shorted_Countries':'Country','Education':'FormalEducation'}, inplace=True)

labels_job = df2_job['JobSatisfaction']
features_job = df2_job.drop('JobSatisfaction', axis=1)

features_job.ConvertedSalary = features_job.ConvertedSalary.fillna(features_job.ConvertedSalary.mean())


dummies_job = pd.get_dummies(features_job)

dum_columns_job = dummies_job.columns


train_features_job, test_features_job, train_labels_job, test_labels_job = train_test_split(dummies_job, labels_job, test_size=0.2, random_state=42)

pca_job = PCA(n_components = 100)
print('PCA Job')
pc = pca_job.fit_transform(dummies_job)


train_PCA = pca_job.transform(train_features_job)
test_PCA= pca_job.transform(test_features_job)


"""
#SVC
lin_clf = svm.LinearSVC()


lin_clf.fit(train_features_job,train_labels_job) 


predict = lin_clf.predict(test_features_job)
"""

"""
#Decision Trees
clf_d = tree.DecisionTreeClassifier()
clf_dncs = tree.DecisionTreeClassifier()
train_features_job_ncs= train_features_job.drop('ConvertedSalary',axis=1)
test_features_job_ncs =  test_features_job.drop('ConvertedSalary',axis=1)

clf_d = clf_d.fit(train_features_job,train_labels_job)
yd_pred=clf_d.predict(test_features_job)

clf_dncs = clf_dncs.fit(train_features_job_ncs,train_labels_job)
yd_predncs=clf_dncs.predict(test_features_job_ncs)
"""
print('Decision Tree')
clf_dPCA = tree.DecisionTreeClassifier()
clf_dPCA = clf_dPCA.fit(train_PCA,train_labels_job)
yd_predPCA=clf_dPCA.predict(test_PCA)
print(accuracy_score(test_labels_job,yd_predPCA))

"""
#Random Forest
clf_rn = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf_rn.fit(train_features_job,train_labels_job)

predict_rn = clf_rn.predict(test_features_job)
accuracy_score(test_labels_job,predict_rn)
"""

def AI_reduce_columns(df):
    AI_columns = ['OpenSource','Country','Employment','FormalEducation','UndergradMajor','CompanySize','DevType','YearsCoding','YearsCodingProf','DatabaseWorkedWith','PlatformWorkedWith','FrameworkWorkedWith','OperatingSystem','Gender','Age','AIDangerous','AIInteresting','AIResponsible','AIFuture']
    df_AI = df[AI_columns]
    df_AI_nans = df_AI.dropna(subset=['AIFuture'])
    return df_AI_nans

df_AI_nans = AI_reduce_columns(df)

country_map = other_categories(df_AI_nans.Country.value_counts(), 400)


df_AI_nans['Shorted_Countries'] = df_AI_nans.Country.map(country_map)

df_AI_nans['Education'] = df_AI_nans.FormalEducation.map(lookup_education)

df2 = df_AI_nans.copy()

del(df2['Country'])

del(df2['FormalEducation'])

df2.rename({'Shorted_Countries':'Country','Education':'FormalEducation'}, inplace=True)


labels = df2['AIFuture']
features = df2.drop('AIFuture', axis=1)


dummies = pd.get_dummies(features)

dum_columns = dummies.columns

train_features, test_features, train_labels, test_labels = train_test_split(dummies, labels, test_size=0.2, random_state=42)

print('SVC for AI')
lin_clf = svm.LinearSVC()
lin_clf.fit(train_features, train_labels)
result = lin_clf.predict(test_features)
print(accuracy_score(test_labels, result))

"""
#Heat map
cm = confusion_matrix(test_labels, result)
ax = sn.heatmap(cm, annot=True, fmt="d")
plt.matshow(cm)
"""

"""
#Random Forest
RandomForestClf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
RandomForestClf.fit(train_features, train_labels)
RandomForestResult = RandomForestClf.predict(test_features)
accuracy_score(test_labels, RandomForestResult)
"""