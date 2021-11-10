#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("/Users/kevinjoo/Documents/df_main.csv")
df.reset_index(drop=True)
df.shape


# In[3]:


import numpy as np
df.replace('None', np.nan, inplace=True)
df.fillna(0, inplace=True)


# In[4]:


df["FG%"] = df[["FG%"]].astype(np.float64)
df["3P%"] = df[["3P%"]].astype(np.float64)
df["2P%"] = df[["2P%"]].astype(np.float64)
df["FT%"] = df[["FT%"]].astype(np.float64)
df["Salary"] = df[["Salary"]].astype(np.float64)
df["Year"] = pd.to_datetime(df["Year"], format="%Y")


# In[5]:


df.info()


# In[6]:


df["Salary"].describe()


# In[7]:


df.drop(df[df["Salary"] < 500000].index, inplace=True)
df.drop(df[df["Salary"] > 12000000].index, inplace=True)


# In[8]:


df.info()


# In[9]:


df["Salary"].describe()


# In[10]:


Y = df["Salary"]
X = df[["FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]]


# In[11]:


X.shape


# In[12]:


Y.shape


# In[144]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[167]:


corr=df.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[ ]:





# In[ ]:





# In[89]:


X2 = df[["3P%", "2P%", "FT%", "ORB", "DRB", "AST", "STL", "BLK", "TOV", "PF"]]


# In[90]:


X_train_val, X_test, y_train_val, y_test = train_test_split(X2, Y, test_size=0.2, random_state=40)


# In[17]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['variables'] = X.columns
vif['vif'] =[variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif


# In[91]:


vif = pd.DataFrame()
vif['variables'] = X2.columns
vif['vif'] =[variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]
vif


# In[92]:


from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression


# In[93]:


lin_reg_est = LinearRegression()


# In[94]:


lin_reg_est.fit(X_train_val, y_train_val)


# In[95]:


y_train_pred = lin_reg_est.predict(X_train_val)


# In[96]:


lin_reg_residuals = y_train_val - y_train_pred

plt.scatter(y_train_pred, lin_reg_residuals, marker="+", alpha=0.5)
plt.plot([0,1], [0, 0])
plt.title("Residuals vs. Predictions")
plt.xlabel("Predictions")
plt.ylabel("Residuals")


# In[97]:


stats.probplot(df["Resid"], dist="norm", plot=plt)
plt.title("Normal Q-Q plot")
plt.show()


# In[98]:


log_y = np.log(Y)
log_model = sm.OLS(log_y, X2)
log_fit = log_model.fit()
log_fit.summary()


# In[99]:


df['log_predict']=log_fit.predict(X2)
df['log_resid']=np.log(df["Salary"])-df['log_predict']


# In[100]:


stats.probplot(df['log_resid'], dist="norm", plot=plt)
plt.title("Log Q-Q plot")
plt.show()


# In[ ]:





# In[102]:


#Simple Linear Regression Model


# In[103]:


linear_model = LinearRegression()
linear_model.fit(X_train_val,y_train_val)
list(zip(X_train_val.columns, linear_model.coef_))


# In[104]:


linear_model.score(X_test,y_test)


# In[105]:


test_set_pred = linear_model.predict(X_test)


# In[106]:


def MAE(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true)) 


# In[107]:


MAE(y_test, test_set_pred)


# In[108]:


r2_score(y_test, test_set_pred)


# In[109]:


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.stats as stats


# In[159]:


def chart(y_pred, y_true, resid, filestr):
    chart = sns.regplot(x=y_pred, y=y_true, ci=False, fit_reg=False, line_kws={'color': 'blue'}, scatter_kws={'s': 5})
    chart.set_xlabel('Predicted Salary')
    chart.set_ylabel('Target Salary')
    chart.plot(y_true, y_true, '--', color='gray')


# In[160]:


chart(test_set_pred, y_test, test_set_pred-y_test, "linear_chart")


# In[161]:


from sklearn.preprocessing import PolynomialFeatures


# In[162]:


poly_features = PolynomialFeatures(2).fit_transform(X_train_val)
linear_model.fit(poly_features,y_train_val)


# In[163]:


linear_model.score(PolynomialFeatures(2).fit_transform(X_test),y_test)


# In[164]:


test_set_pred = linear_model.predict(PolynomialFeatures(2).fit_transform(X_test))


# In[165]:


MAE(y_test, test_set_pred)


# In[166]:


r2_score(y_test, test_set_pred)


# In[ ]:





# In[ ]:





# In[147]:


from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.metrics import r2_score
import numpy as np


# In[149]:


X_train_val, X_test, y_train_val, y_test = train_test_split(X2, Y, test_size=0.2,random_state=40)


# In[150]:


std = StandardScaler()
std.fit(X_train_val.values)


# In[151]:


X_tr = std.transform(X_train_val.values)


# In[152]:


X_te = std.transform(X_test.values)


# In[153]:


lasso = Lasso(alpha = 10000)


# In[155]:


lasso.fit(X_tr, y_train_val)


# In[156]:


test_r_squared = lasso.score(X_te, y_test)


# In[157]:


print(test_r_squared)


# In[158]:


print(list(zip(X_train_val.columns, lasso.coef_)))


# In[ ]:





# In[128]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


# In[129]:


alphas = 10**np.linspace(-2,2,200)


# In[130]:


lasso_model = LassoCV(alphas = alphas, cv=5)


# In[131]:


lasso_model.fit(X_train, y_train_val)


# In[132]:


r_squared_train = lasso_model.score(X_train, y_train_val)


# In[133]:


r_squared_test = lasso_model.score(X_test, y_test)


# In[134]:


alpha = lasso_model.alpha_


# In[135]:


print(r_squared_train)
print(r_squared_test)
print(alpha)


# In[ ]:





# In[136]:


from sklearn.linear_model import lars_path
import matplotlib.pyplot as plt


# In[137]:


alphas = 10**np.linspace(-2,2,200)


# In[138]:


alphas, _, coefs = lars_path(X_train, y_train_val.values, method='lasso')


# In[139]:


xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]


# In[140]:


plt.figure(figsize=(10,10))
plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.legend(X_train_val.columns)
plt.show()


# In[ ]:





# In[168]:


from sklearn.model_selection import RepeatedKFold


# In[172]:


kf = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)


# In[173]:


cv_lm_r2s, cv_lm_scale_r2s = [], [] 
for train_ind, val_ind in kf.split(X2,Y):
    X_train_val, y_train_val = X2.iloc[train_ind], Y.iloc[train_ind]
    X_val, y_val = X2.iloc[val_ind], Y.iloc[val_ind] 


# In[174]:


lm = LinearRegression()
lm.fit(X_train_val, y_train_val)
cv_lm_r2s.append(lm.score(X_val, y_val))


# In[175]:


print('Simple regression scores: ', cv_lm_r2s)
print(f'Simple mean cv r^2: {np.mean(cv_lm_r2s):.3f} +- {np.std(cv_lm_r2s):.3f}')

