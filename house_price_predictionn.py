##################################################################
# House Price Prediction Model
#################################################################

## Problem Statement
# Predicting house prices for different types of houses based on their features using machine learning.

## Dataset Overview
# The dataset contains information on 79 explanatory variables with 2919 rows and sale prices of houses located in Ames, Iowa.
# The dataset is from a Kaggle competition, and it's divided into two separate CSV files: train and test.
# The train dataset includes the sale prices, while the test dataset has the sale prices left blank, requiring
# you to predict them.


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import combinations
from scipy.stats import chi2_contingency
from scipy.stats import skew

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_validate

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

import warnings
warnings.simplefilter("ignore")

################################################
# TASK 1 - EXPLORATORY DATA ANALYSIS (EDA)
################################################

# Read and Combine Train and Test Datasets
##################################################
data1 = pd.read_csv('Datasets/train_house_pred.csv')
data2 = pd.read_csv('Datasets/test_house_pred.csv')

df = pd.concat([data1, data2], ignore_index=True)

# Convert Column Names to Lowercase
###########################################
df.columns = [col.lower() for col in df.columns]

# Check DataFrame Information
#############################
def check_df(dataframe, head=5):
    print('################# Columns ################# ')
    print(dataframe.columns)
    print('################# Types  ################# ')
    print(dataframe.dtypes)
    print('##################  Head ################# ')
    print(dataframe.head(head))
    print('#################  Shape ################# ')
    print(dataframe.shape)
    print('#################  NA ################# ')
    print(dataframe.isnull().sum())
    print('#################  Quantiles ################# ')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99]).T)

check_df(df)


# Capture Numerical and Categorical Variables
#############################################
def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Perform Necessary Data Adjustments
####################################
df['mssubclass'] = df['mssubclass'].astype(object)

df['yrsold'] = df['yrsold'].astype(int)

df = df.drop('id',axis=1)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    print(col, df[col].unique())

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Explore the Distribution of Numerical and Categorical Variables
##################################################################

# Categorical variables:
########################
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    if df[col].dtypes == 'bool':
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)

    else:
        cat_summary(df, col, plot=True)


# Numerical variables:
######################
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot = True)


# Analyze Categorical Variables in Relation to the Target Variable
###################################################################

def target_summary_with_cat(dataframe, target, categorical_col):
   print(pd.DataFrame({'TARGET_MEAN': dataframe.groupby(categorical_col)[target].mean()}))

for col in cat_cols:
    target_summary_with_cat(df, 'saleprice', col)

# Analysis of Correlation
######################################
corr = df[num_cols].corr()
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show(block=True)

def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show(block=True)
    return drop_list

high_correlated_cols(df[num_cols], plot=False)

# Analyzse Outliers
###########################################
# Setting outlier threshold value (by using the IQR method):
############################################################
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Define a function to check for outliers in a column
#####################################################
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    if col != "saleprice":
      print(col, check_outlier(df, col))

# Suppression of outliers
######################################################
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if col != "saleprice":
        replace_with_thresholds(df,col)

# Analyze Missing Values
########################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

#               n_miss  ratio
# poolqc          2909 99.660
# miscfeature     2814 96.400
# alley           2721 93.220
# fence           2348 80.440
# saleprice       1459 49.980 # target
# fireplacequ     1420 48.650
# ...

# Null values in some variables indicate that the house does not have that feature
##################################################################################
no_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]

# Filling the gaps in the columns with the expression "No"
for col in no_cols:
    df[col].fillna("No", inplace=True)


# Define a function to fill missing values with mean (numerical) and mode (categorical)
#######################################################################################
def fill_missing(df, target_col):
    filled_df = df.copy()

    for column in filled_df.columns:
        if column != target_col and filled_df[column].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(filled_df[column]):
                column_mean = filled_df[column].mean()
                filled_df[column].fillna(column_mean, inplace=True)
            elif filled_df[column].dtype == 'O':
                column_mode = filled_df[column].mode()[0]
                filled_df[column].fillna(column_mode, inplace=True)

    return filled_df

df = fill_missing(df, target_col='saleprice')
df.isnull().sum()


################################################
# TASK 2: FEATURE ENGINEERING
################################################

# Label encoder for ordinal variables
#####################################
def label_encoder(dataframe, col):
    labelencoder = LabelEncoder()
    for c in col:
      dataframe[c] = labelencoder.fit_transform(dataframe[c])
    return dataframe


le_col = ['lotshape', 'overallqual', 'overallcond', 'exterqual', 'extercond',
       'bsmtqual', 'bsmtexposure', 'bsmtfintype1', 'bsmtfintype2',
       'heatingqc', 'kitchenqual']

df = label_encoder(df, le_col)

df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Check and Handle High Correlations Among Numerical Variables
####################3#########################################
def high_correlated_cols(dataframe, corr_th=0.7):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))

    correlated_pairs = []

    for col in upper_triangle_matrix.columns:
        high_corr_cols = upper_triangle_matrix.index[upper_triangle_matrix[col] > corr_th].tolist()
        for high_corr_col in high_corr_cols:
            correlation = corr.loc[col, high_corr_col]
            correlated_pairs.append((col, high_corr_col, correlation))

    return correlated_pairs

correlated_pairs = high_correlated_cols(df, corr_th=0.7)

for pair in correlated_pairs:
    print(f"High Correlation: {pair[0]} - {pair[1]} (Correlation Coefficient: {pair[2]:.3f})")

df = df.drop(columns=['garagecars', 'totrmsabvgrd',])

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Create New Features
#####################
df["NEW_1st*GrLiv"] = df["1stflrsf"] * df["grlivarea"]

df["NEW_Garage*GrLiv"] = (df["garagearea"] * df["grlivarea"])

df["TotalQual"] = df[["overallqual", "overallcond", "exterqual", "extercond", "bsmtcond", "bsmtfintype1",
                      "bsmtfintype2", "heatingqc", "kitchenqual", "functional", "garagequal", "garagecond",]].sum(axis=1,numeric_only=True)


# Total Floor
df["NEW_TotalFlrSF"] = df["1stflrsf"] + df["2ndflrsf"] # 32

# Total Finished Basement Area
df["NEW_TotalBsmtFin"] = df.bsmtfintype1 + df.bsmtfintype2 # 56

# Porch Area
df["NEW_PorchArea"] = df.openporchsf + df.enclosedporch + df.screenporch + df["3ssnporch"] + df.wooddecksf # 93

# Total House Area
df["NEW_TotalHouseArea"] = df.NEW_TotalFlrSF + df.totalbsmtsf # 156

df["NEW_TotalSqFeet"] = df.grlivarea + df.totalbsmtsf # 35

# Lot Ratio
df["NEW_LotRatio"] = df.grlivarea / df.lotarea # 64

df["NEW_RatioArea"] = df.NEW_TotalHouseArea / df.lotarea # 57

df["NEW_GarageLotRatio"] = df.garagearea / df.lotarea # 69

# MasVnrArea
df["NEW_MasVnrRatio"] = df.masvnrarea / df.NEW_TotalHouseArea # 36

# Dif Area
df["NEW_DifArea"] = (df.lotarea - df["1stflrsf"] - df.garagearea - df.NEW_PorchArea - df.wooddecksf) # 73


df["NEW_OverallGrade"] = df["overallqual"] * df["overallcond"] # 61

df["NEW_Restoration"] = df.yearremodadd - df.yearbuilt # 31

df["NEW_HouseAge"] = df.yrsold - df.yearbuilt # 73

df["NEW_RestorationAge"] = df.yrsold - df.yearremodadd # 40

df["NEW_GarageAge"] = df.garageyrblt - df.yearbuilt # 17

df["NEW_GarageRestorationAge"] = np.abs(df.garageyrblt - df.yearremodadd) # 30

df["NEW_GarageSold"] = df.yrsold - df.garageyrblt # 48

#drop_list = ["street", "alley", "landcontour", "utilities", "landslope","heating", "poolqc", "miscfeature", "neighborhood"]
#df.drop(drop_list, axis=1, inplace=True)

# Rare Encoder
###############
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "saleprice", cat_cols)

df = df.drop(columns=['3ssnporch', 'poolarea', 'miscval', 'lowqualfinsf'])

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def rare_encoder(dataframe, rare_perc, col=None):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
        print(f"Column: {col}")
        print(tmp)
        print("=" * 30)

    return temp_df

df = rare_encoder(df, 0.05)

df.head()

for col in cat_cols:
    print(col, df[col].unique())

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Label Encoder - Binary Columns
#################################
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# One-hot Encoder
#################
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if (10 >= df[col].nunique() > 2) & (col not in le_col) and (col not in binary_cols)]

df = one_hot_encoder(df, ohe_cols)


# Standardization
##################
scaler = StandardScaler()

num_cols = [col for col in df.columns if col != 'saleprice']

df[num_cols] = scaler.fit_transform(df[num_cols])


#######################################
# TASK 3 - MODELLING
#######################################

# Splitting Train and Test Data
################################
train_df = df[df['saleprice'].notnull()]
test_df = df[df['saleprice'].isnull()]

y = train_df['saleprice']
X = train_df.drop(["saleprice"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor(verbose=-1)),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

# Results :
#RMSE: 1895700611790816.5 (LR)
#RMSE: 34461.2211 (Ridge)
#RMSE: 34640.016 (Lasso)
#RMSE: 30957.4506 (ElasticNet)
#RMSE: 35695.8306 (KNN)
#RMSE: 38182.963 (CART)
#RMSE: 28291.9813 (RF)
#RMSE: 81133.0365 (SVR)
#RMSE: 26177.0563 (GBM)
#RMSE: 28669.0579 (XGBoost)
#RMSE: 28198.149 (LightGBM)
#RMSE: 25168.6781 (CatBoost)


#################################
# 4. HYPERPARAMETER OPTIMIZATION
#################################
#-- > LightGBM Model is chosen for hyperparameter optimization
lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)
lgbm_gs_best.best_params_
# --> 'learning_rate': 0.01, 'n_estimators': 500}
final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

RMSE = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))

#RMSE: 27481.75868226194

# Feature importance for CatBoost
#################################
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:20])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm_model, X)












































































