#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


# In[2]:


# set number of max rows and columns
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


# In[3]:


#Reading in data
df = pd.read_csv('./Data/PHY_TRAIN.csv')
df.head()
df = df.drop('exampleid', axis=1)


# In[4]:


# get the skew of the features
df_skew = pd.DataFrame(df.skew(), columns=['skew'])
df_skew.head(5)


# In[5]:


# get summary stats of features and add skew to table
df_summary = df.describe().transpose()
df_summary = df_summary[['mean', 'std', 'min', 'max']]
df_summary = pd.concat([df_summary, df_skew], axis=1)
df_summary.head()


# In[6]:


# calculate the missing value proportions for each variable
na_sum = df.isna().sum()
df_summary['miss_rate'] = na_sum / 50000
df_summary.to_csv('./Data/summary_stats.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


# split df feature set and predictor set
X = df.iloc[:, 1:] 
y = df.iloc[:, 0]


# In[8]:


# creating missing value indicator dataframe
cols = list(X.columns)
X_mvi = X[cols].isnull().astype(int).add_prefix('M_') #Creating df for MVI
X_with_mvi = pd.concat([X.fillna(X.mean()), X_mvi], axis=1) #Joining MVI df with real df and filling na values with column mean
#print (X_with_mvi)


# In[9]:


#Checking dataset for collinearity
X_imp = X.fillna(X.mean())

# create a correlation matrix
corr = X_imp.corr().fillna(0)

# plot the graph
fig = plt.figure(figsize=(25, 25))
plt.matshow(corr, fignum=fig.number)
plt.xticks(range(corr.shape[1]), corr.columns, fontsize=14, rotation=90)
plt.yticks(range(corr.shape[1]), corr.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Feature Correlation Matrix', fontsize=26)
plt.savefig('./Data/correlation_heatmap.png', dpi=100);


# Strong collinearity is present between several variables. Some will need to be removed in order to model the data. 

# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


'''
#Found a code to caluculate VIF of all columns and remove any greater than 5. Returns DF with uncorrelated columns
#Source: https://stats.stackexchange.com/a/253620/19676

from statsmodels.stats.outliers_influence import variance_inflation_factor    

def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

X_vif_red = calculate_vif_(X_imp)
''';


# In[12]:


'''
#Checking correlations
X_vif_red.corr()
#X_vif_red.isna().sum()

#Nan values in correlation matrix are unexpected. Removing them.

#ONLY RUN THIS ONCE
X_vif_red = X_vif_red.drop(['feat29','feat47','feat48','feat49','feat50','feat51','feat55'], axis=1)

#print (np.linalg.det(X_vif_red)) #This having an error could be an issue

#Viewing highest correlations in data
c = X_vif_red.corr().abs()
print(c.shape)
s = c.unstack()
so = s.sort_values(kind="quicksort")
print (so[-60:-50])
''';


# In[13]:


# run this line of code to load in the saved feature-parsed dataset
#X_vif_red.to_csv('./Data/df_no_corr.csv', index=False)    # run to save above codeblock result
X_vif_red = pd.read_csv('./Data/df_no_corr.csv')    # run to load above codebloack result


# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


# Running Logistic regression model from stats models package on all remaining variables (VIF<5 and correlation of nan removed)
X_const = sm.add_constant(X_vif_red)
classifier = sm.Logit(y, X_const)
log_result = classifier.fit(method='newton')
print(log_result.summary())


# In[15]:


# list out the features with p-value below 0.05
keep_features = ['const', 'feat4', 'feat8', 'feat12', 'feat13', 'feat14', 'feat15', 'feat20', 'feat31', 'feat39', 'feat40', 'feat42', 'feat56', 'feat63', 'feat66', 'feat70', 'feat71', 'feat75']


# In[16]:


#Calculating AUC and performance metrics

# get result probabilities
y_pred_log = log_result.predict(X_const)

# using a threshold of 0.5, classify the predictions as 0 or 1 based on the result probabilities
y_pred_binary_log = pd.DataFrame(y_pred_log, columns=['y_pred'])
y_pred_binary_log['prediction'] = np.where(y_pred_binary_log.y_pred >= 0.5, 1, 0)

# training AUC
log_auc = roc_auc_score(y, y_pred_binary_log.prediction)
print("AUC of Initial Logistic Regression model is", log_auc)

# calculate performance metrics
print('Accuracy: ', accuracy_score(y, y_pred_binary_log.prediction))
print('Precision; ', precision_score(y, y_pred_binary_log.prediction))
print('Recall: ', recall_score(y, y_pred_binary_log.prediction))
print('F1: ', f1_score(y, y_pred_binary_log.prediction))


# In[17]:


# confusion matrix for initial log regression
tn, fp, fn, tp = confusion_matrix(y, y_pred_binary_log.prediction).ravel()
cm_log = pd.DataFrame(confusion_matrix(y, y_pred_binary_log.prediction))
print('Confusion Matrix:\n',cm_log, '\n')
print('True Positive: ', tp)
print('True Negative: ', tn)
print('False Negative: ', fn)
print('False Positive: ', fp)


# In[18]:


# create roc curve for logistic regression

# get false positive rate, true positive rate and thresholds for training data
fpr_train_log, tpr_train_log, thresholds_train_log = roc_curve(y, y_pred_log)

# plot the roc curve
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(fpr_train_log, tpr_train_log, label='Log Regr. Initial')
ax.plot(np.linspace(0,1,25), np.linspace(0,1,25), 'k--')
plt.title('ROC Curve for Logistic Regression with feature VIF > 5 Removed', fontsize=28)
plt.xlabel('False Positive Rate', fontsize=24)
plt.ylabel('True Positive Rate', fontsize=24)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
plt.savefig('./Data/roc_logistic_regression_vif.png', dpi=100);


# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


# remove features from with p-values greater than 0.05 and perform logistic regression again
X_reduce = X_const[keep_features]

# Running Logistic regression model from stats models package on reduced feature list
classifier = sm.Logit(y, X_reduce)
result_log_reduce = classifier.fit(method='newton')
print(result_log_reduce.summary())


# In[20]:


#Calculating AUC and performance metrics

# get result probabilities
y_pred_log_reduce = result_log_reduce.predict(X_reduce)

# using a threshold of 0.5, classify the predictions as 0 or 1 based on the result probabilities
y_pred_binary_log_reduce = pd.DataFrame(y_pred_log_reduce, columns=['y_pred'])
y_pred_binary_log_reduce['prediction'] = np.where(y_pred_binary_log_reduce.y_pred >= 0.5, 1, 0)

# training AUC
log_red_auc = roc_auc_score(y, y_pred_binary_log_reduce.prediction)
print("AUC of Logistic Regression model with reduced deatures is", log_red_auc)

# calculate performance metrics
print('Accuracy: ', accuracy_score(y, y_pred_binary_log_reduce.prediction))
print('Precision; ', precision_score(y, y_pred_binary_log_reduce.prediction))
print('Recall: ', recall_score(y, y_pred_binary_log_reduce.prediction))
print('F1: ', f1_score(y, y_pred_binary_log_reduce.prediction))


# In[21]:


# confusion matrix for reduced feature log regression
tn, fp, fn, tp = confusion_matrix(y, y_pred_binary_log_reduce.prediction).ravel()
cm_log = pd.DataFrame(confusion_matrix(y, y_pred_binary_log_reduce.prediction))
print('Confusion Matrix:\n',cm_log, '\n')
print('True Positive: ', tp)
print('True Negative: ', tn)
print('False Negative: ', fn)
print('False Positive: ', fp)


# In[22]:


# create roc curve for logistic regression with reduced features

# get false positive rate, true positive rate and thresholds for training data
fpr_train_log_reduce, tpr_train_log_reduce, thresholds_train_log_reduce = roc_curve(y, y_pred_log_reduce)

# plot the roc curve
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(fpr_train_log_reduce, tpr_train_log_reduce, color='orange', label='Log Regr. Reduced')
ax.plot(np.linspace(0,1,25), np.linspace(0,1,25), 'k--')
plt.title('ROC Curve for Logistic Regression with Reduced Features', fontsize=28)
plt.xlabel('False Positive Rate', fontsize=24)
plt.ylabel('True Positive Rate', fontsize=24)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
plt.savefig('./Data/roc_logistic_regression_reduced.png', dpi=100);


# In[ ]:





# In[ ]:





# In[ ]:





# ### Logistic regression with two-way interaction

# In[23]:


# identify five features with the highest predictive power (aka highest z-scores)
top_features = ['feat4', 'feat8', 'feat13', 'feat14', 'feat75']


# In[24]:


# add all possible interactions to dataset
X_interact = X_reduce.copy()
for i in top_features:
    for j in top_features:
        if int(i[4:]) < int(j[4:]):
            X_interact[i+'x'+j] = X_interact[i] * X_interact[j]

X_interact.iloc[:1,-15:]


# In[25]:


# Running Logistic regression model from stats models package on reduced feature list
classifier = sm.Logit(y, X_interact)
log_interact_result = classifier.fit(method='newton')
print(log_interact_result.summary())


# In[26]:


#Calculating AUC and performance metrics

# get result probabilities
y_pred_log_interact = log_interact_result.predict(X_interact)

# using a threshold of 0.5, classify the predictions as 0 or 1 based on the result probabilities
y_pred_binary_log_interact = pd.DataFrame(y_pred_log_interact, columns=['y_pred'])
y_pred_binary_log_interact['prediction'] = np.where(y_pred_binary_log_interact.y_pred >= 0.5, 1, 0)

# training AUC
log_interact_auc = roc_auc_score(y, y_pred_binary_log_interact.prediction)
print("AUC of Logistic Regression with Interaction Model is", log_interact_auc)

# calculate performance metrics
print('Accuracy: ', accuracy_score(y, y_pred_binary_log_interact.prediction))
print('Precision; ', precision_score(y, y_pred_binary_log_interact.prediction))
print('Recall: ', recall_score(y, y_pred_binary_log_interact.prediction))
print('F1: ', f1_score(y, y_pred_binary_log_interact.prediction))


# In[27]:


# confusion matrix for feature interaction log regression
tn, fp, fn, tp = confusion_matrix(y, y_pred_binary_log_interact.prediction).ravel()
cm_log = pd.DataFrame(confusion_matrix(y, y_pred_binary_log_interact.prediction))
print('Confusion Matrix:\n',cm_log, '\n')
print('True Positive: ', tp)
print('True Negative: ', tn)
print('False Negative: ', fn)
print('False Positive: ', fp)


# In[28]:


# create roc curve for logistic regression with interaction

# get false positive rate, true positive rate and thresholds for training data
fpr_train_log_interact, tpr_train_log_interact, thresholds_train_log_interact = roc_curve(y, y_pred_log_interact)

# plot the roc curve
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(fpr_train_log_interact, tpr_train_log_interact, color='purple', label='Logistic Regr. with Interaction')
ax.plot(np.linspace(0,1,25), np.linspace(0,1,25), 'k--')
plt.title('ROC Curve for Logistic Regression with Interaction Features', fontsize=28)
plt.xlabel('False Positive Rate', fontsize=24)
plt.ylabel('True Positive Rate', fontsize=24)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
plt.savefig('./Data/roc_logistic_regression_interact.png', dpi=100);


# In[ ]:





# In[ ]:





# In[ ]:





# In[29]:


#Random forest modeling
rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
rf.fit(X_imp, y)


# In[30]:


#Calculating AUC and performance metrics

# get result probabilities
y_pred_rf = rf.predict(X_imp)
y_pred_prob_rf = pd.DataFrame(rf.predict_proba(X_imp))

# using a threshold of 0.5, classify the predictions as 0 or 1 based on the result probabilities
#y_pred_binary = pd.DataFrame(y_pred, columns=['y_pred'])
#y_pred_binary['prediction'] = np.where(y_pred_binary.y_pred >= 0.5, 1, 0)

# calculate AUC
rf_auc = roc_auc_score(y, y_pred_rf)
print("AUC of Random Forest model is", rf_auc)

# calculate performance metrics
print('Accuracy: ', accuracy_score(y, y_pred_rf))
print('Precision; ', precision_score(y, y_pred_rf))
print('Recall: ', recall_score(y, y_pred_rf))
print('F1: ', f1_score(y, y_pred_rf))


# In[31]:


# confusion matrix for random forest
tn, fp, fn, tp = confusion_matrix(y, y_pred_rf).ravel()
cm_log = pd.DataFrame(confusion_matrix(y, y_pred_rf))
print('Confusion Matrix:\n',cm_log, '\n')
print('True Positive: ', tp)
print('True Negative: ', tn)
print('False Negative: ', fn)
print('False Positive: ', fp)


# In[32]:


# create roc curve for random forest

# get false positive rate, true positive rate and thresholds for training data
fpr_train_rf, tpr_train_rf, thresholds_train_rf = roc_curve(y, y_pred_prob_rf[1])

# plot the roc curve
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(fpr_train_rf, tpr_train_rf, label='Random Forest', color='green')
ax.plot(np.linspace(0,1,25), np.linspace(0,1,25), 'k--')
plt.title('ROC Curve for Random Forest', fontsize=28)
plt.xlabel('False Positive Rate', fontsize=24)
plt.ylabel('True Positive Rate', fontsize=24)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
plt.savefig('./Data/roc_random_forest.png', dpi=100);


# In[ ]:





# In[ ]:





# In[ ]:





# In[33]:


# gradient boosting model using sklearn
gb = GradientBoostingClassifier(n_estimators=100, random_state = 42)
gb.fit(X_imp, y)

# get prediction values and probabilities
y_pred_gb = gb.predict(X_imp)
y_pred_prob_gb = pd.DataFrame(gb.predict_proba(X_imp))

# calculate AUC and performance metrics
gb_auc_score = roc_auc_score(y, y_pred_gb)
print('AUC of gradient boosting model is', gb_auc_score)
print('Accuracy: ', accuracy_score(y, y_pred_gb))
print('Precision; ', precision_score(y, y_pred_gb))
print('Recall: ', recall_score(y, y_pred_gb))
print('F1: ', f1_score(y, y_pred_gb))


# In[34]:


# confusion matrix for gradient boosting
tn, fp, fn, tp = confusion_matrix(y, y_pred_gb).ravel()
cm_log = pd.DataFrame(confusion_matrix(y, y_pred_gb))
print('Confusion Matrix:\n',cm_log, '\n')
print('True Positive: ', tp)
print('True Negative: ', tn)
print('False Negative: ', fn)
print('False Positive: ', fp)


# In[35]:


# create roc curve for gradient boosting

# get false positive rate, true positive rate and thresholds for training data
fpr_train_gb, tpr_train_gb, thresholds_train_gb = roc_curve(y, y_pred_prob_gb[1])

# plot the roc curve
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(fpr_train_gb, tpr_train_gb, color='red', label='Gradient Boosting')
ax.plot(np.linspace(0,1,25), np.linspace(0,1,25), 'k--')
plt.title('ROC Curve for Gradient Boosting', fontsize=28)
plt.xlabel('False Positive Rate', fontsize=24)
plt.ylabel('True Positive Rate', fontsize=24)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
plt.savefig('./Data/roc_gradient_boosting.png', dpi=100);


# In[ ]:





# In[ ]:





# In[ ]:





# In[36]:


# plot all curves on same graph
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(fpr_train_gb, tpr_train_gb, color='red', label='Gradient Boosting')
ax.plot(fpr_train_rf, tpr_train_rf, label='Random Forest', color='green')
ax.plot(fpr_train_log_interact, tpr_train_log_interact, color='purple', label='Logistic Regr. with Interaction')
ax.plot(fpr_train_log_reduce, tpr_train_log_reduce, color='orange', label='Log Regr. Reduced')
ax.plot(fpr_train_log, tpr_train_log, label='Log Regr. Initial')
ax.plot(np.linspace(0,1,25), np.linspace(0,1,25), 'k--')
plt.legend(fontsize=20)
plt.title('ROC Curves of All Models', fontsize=28)
plt.xlabel('False Positive Rate', fontsize=24)
plt.ylabel('True Positive Rate', fontsize=24)
plt.savefig('./Data/roc_combined.png', dpi=100);


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




