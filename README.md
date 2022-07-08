import warnings
warnings.filterwarnings('ignore')

# 데이터 읽기를 위한 라이브러리
import numpy as np
np.random.seed(0)
import pandas as pd
import gc, os, time
import scipy as sp
from pandas import DataFrame, Series
from datetime import datetime, date, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 탐색적 데이터 분석을 위한 라이브러리
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, norm, probplot, boxcox

# 모델링을 위한 라이브러리
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold

import pickle

# 학습용 데이터 
train_sensor = pd.read_csv('train_sensor.csv')
train_quality = pd.read_csv('train_quality.csv')

# 평가용 데이터 
predict_sensor = pd.read_csv('predict_sensor.csv')


def make_dataset(X, y=None):
    
    # -----------------------------------
    # train_sensor (X 인자)
    # -----------------------------------
    ''' column을 param_alias 로만 pivot table 만들기. '''
    df_X = X.copy()
    df_X = df_X.sort_values(by='end_time',ascending=True)
    df_X['step_id'] = df_X['step_id'].apply(lambda x: str(x).zfill(2))
    # step_id 와 param_alias 를 결합한 임시 컬럼 step_param 을 생성합니다. ex. 17_EPD_para4
    df_X['step_param'] = df_X[['step_id', 'param_alias']].apply(lambda x: '_'.join(x), axis=1)
    df_X_tmp = df_X.pivot_table(index = ['module_name','key_val'], columns = 'step_param', values='mean_val', aggfunc='sum')
    # 데이터 통합을 위해 인덱스를 key_val 로 재설정합니다. 
    df_X_tmp = df_X_tmp.reset_index(level=[0, 1])
    df_X_tmp.set_index('key_val', inplace=True)

    # -----------------------------------
    # 시간 데이터 
    # -----------------------------------
    ''' step별 end_time을 column으로 pivot table 만들기 '''
    df_X['end_time_tmp'] = df_X.apply(lambda x: x['step_id'] + '_end_time', axis=1)
    df_X['end_time'] = pd.to_datetime(df_X['end_time'])
    # end_time 은 센서 데이터가 각 para 별로 서버에 도달한 시간으로 스텝 내 오차가 발생할 수 있습니다. 동일 스텝 구간내 공정 완료 시간이 다른 경우, min 함수를 사용하여 최초 수집된 time을 가져옵니다.
    df_time_tmp = df_X.pivot_table(index = ['key_val'], columns = 'end_time_tmp', values='end_time', aggfunc=lambda x : min(x.unique()))
    df_time_tmp = df_time_tmp.reset_index()
    df_time_tmp.set_index('key_val', inplace=True)

    # -----------------------------------
    # train_quality (y 인자)
    # -----------------------------------

    if y is None : # 평가용 데이터 
      
        col_target = []
        col_idx = ['module_name', 'key_val']
        df_complete = pd.concat([df_X_tmp, df_time_tmp], axis=1).reset_index()
    
    else : # 학습용 데이터 
        df_y = y.copy()
        df_y.set_index('key_val', inplace=True)
      
        col_target = ['y']
        col_idx = ['module_name', 'key_val', 'end_dt_tm']
      
        # 센서 데이터, 시간데이터, 품질지표에 대하여 인덱스(key_val)기준으로 데이터프레임을 통합합니다.
        df_complete = pd.concat([df_X_tmp, df_time_tmp, df_y], axis=1).reset_index()

        # 컬럼 이름을 변경합니다.  
        df_complete.rename(columns={'msure_val':'y'}, inplace=True)


    # 컬럼 순서를 정렬합니다. 
    col_feats = df_X['step_param'].unique().tolist()
    col_feats.sort()
    col_time = [s for s in df_complete.columns.tolist() if "_end_time" in s]
    col_all = col_idx + col_target + col_feats + col_time
    df_complete = df_complete[col_all]
    # 처음 step이 시작된 시점을 기준으로 다시 정렬(APC value를 먹고 들어가는 값을 기준으로 정렬하고 싶었음.)
    df_complete = df_complete.set_index(['module_name','key_val','04_end_time']).sort_index(level=[0,2,1],ascending=True).reset_index()
    df_complete = df_complete[col_all]
    
    # 컬럼을 소문자로 변경합니다. 
    df_complete.columns = df_complete.columns.str.lower()

    return df_complete


# 학습용 데이터 
train = make_dataset(train_sensor, train_quality)
# 평가용 데이터 
predict = make_dataset(predict_sensor)


2.1 Target Distribution

eda_df = train.copy()

fig = plt.figure(figsize = (15, 40))
for i,name in enumerate(eda_df['module_name'].unique().tolist()):
    ax = plt.subplot(10, 5, i+1)
    im = eda_df.loc[eda_df.module_name == name, 'y'].values
    mean = im.mean().round(3)
    std = im.std().round(3)
    skew = (3*(mean - np.median(im))/im.std()).round(3)
    if skew >= 1.0:
        plt.hist(im, alpha = 0.7, bins = 50, color = 'red')
    elif skew <= -1.0:
        plt.hist(im, alpha = 0.7, bins = 50, color = 'blue')
    else:
        plt.hist(im, alpha = 0.7, bins = 50, color = 'gray')
    plt.title(f'{name}', color='white')
    plt.xticks([],color='white')
    plt.yticks([],color='white')
    plt.xlabel('')
    plt.ylabel('')
    plt.text(0.35, 0.9, f'mean : {mean}',  ha='left', va='center', transform=ax.transAxes)
    plt.text(0.35, 0.8, f'std : {std}',  ha='left', va='center', transform=ax.transAxes)
    plt.text(0.35, 0.7, f'skew : {skew}',  ha='left', va='center', transform=ax.transAxes)
    
또한, 챔버별로 target skewness가 다릅니다.

positive skew를 보이는 건물(빨간색 histogram)도 있고, negative skew를 보이는 건물(파란색 histogram)도 있습니다. 

다행히, 1.5이상 혹은 -1.5 이하의 큰 편향성을 가지는 건물을 보이지 않으나, 

EQ5_PM6(-1.438), EQ6_PM5(-1.229), EQ11_PM1(-1.017)등의 건물에서 다소 높은 skewness가 관측됨을 확인할 수 있습니다.

target이 편향성을 가지는 경우 모델 성능에 악영향을 주므로, 일단 편향성을 최대한 줄이기 위하여 모든 건물들의 target 값에 대해 log transformation 을 수행하겠습니다.

fig = plt.figure(figsize = (15, 40))
for i,name in enumerate(eda_df['module_name'].unique().tolist()):
    ax = plt.subplot(10, 5, i+1)
    im = np.log(1 + eda_df.loc[eda_df.module_name == name, 'y'].values)
    mean = im.mean().round(3)
    std = im.std().round(3)
    skew = (3*(mean - np.median(im))/im.std()).round(3)
    if skew >= 1.0:
        plt.hist(im, alpha = 0.7, bins = 50, color = 'red')
    elif skew <= -1.0:
        plt.hist(im, alpha = 0.7, bins = 50, color = 'blue')
    else:
        plt.hist(im, alpha = 0.7, bins = 50, color = 'gray')
    plt.title(f'{name}', color='white')
    plt.xticks([],color='white')
    plt.yticks([],color='white')
    plt.xlabel('')
    plt.ylabel('')
    plt.text(0.5, 0.9, f'skew : {skew}',  ha='left', va='center', transform=ax.transAxes)
    
2.2 QUALITY DEGREE OF EACH CHAMBER IN RELATION TO DATETIME

eda_df['weekday'] = pd.to_datetime(eda_df['04_end_time']).dt.weekday
eda_df['hour'] = pd.to_datetime(eda_df['04_end_time']).dt.hour
    
# energy usage of each building ~ weekday, hour
fig = plt.figure(figsize = (15, 40))
for i,module in enumerate(eda_df['module_name'].unique().tolist()):
    df = eda_df[eda_df.module_name == module]
    df = df.groupby(['weekday','hour'])['y'].mean().reset_index().pivot('weekday', 'hour', 'y')
    plt.subplot(10, 5, i+1)
    sns.heatmap(df)
    plt.title(f'{module}',color='white')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(color='white')
    plt.yticks([],color='white')    

# -----------------------------------
# 3 장 EDA 분석에 필요한 변수를 선언합니다.
# -----------------------------------

# 센서 컬럼과 날짜 컬럼을 정의합니다. 
col_sensor = df_eda.iloc[:, 4:-7].columns.tolist() 
col_time = df_eda.filter(regex='end').columns.tolist() 

assert len(col_sensor) == 665
assert len(col_time) == 8 

# 3.4절 공정 소요시간 분석에 필요한 변수를 정의합니다. 
lst_steps = ['04','06','12','13','17','18', '20']
lst_stepsgap = ['0406','0612','1213','1317','1718','1820']

''' step별로 fdc para명 따로 수집 '''
lst_sensors = []
for step in lst_steps:
    _ = [col for col in col_sensor if col[:2] == step]
    lst_sensors.append(_)

sensors_nm = list(map(lambda x: x[3:], lst_sensors[0]))

# 시간과 관련한 분석을 진행하기 위하여 날짜형으로 변환합니다. 
df_eda[col_time] = df_eda[col_time].apply(pd.to_datetime)


idx = 'module_name'
# 모듈의 고유값과 고유값의 갯수를 구합니다.
module_unq = df_eda[idx].unique()
module_nunq = df_eda[idx].nunique()

print(f'***')
print(f'No. of Module : {module_nunq}')
print(f'{module_unq}')

# 모듈 이름의 상위 집계단위를 나타내는 module_name_eq 범주형 변수를 생성하는 함수입니다.
def gen_cate_feats(df):
    df['module_name_eq'] = df['module_name'].apply(lambda x : x.split('_')[0])
    return df 
   
df_eda = gen_cate_feats(df_eda)
sns.boxplot(x='module_name_eq', y='y', data=df_eda)

3.4 전체 공정 소요시간에 따른 타깃 변수의 변화
전체 및 개별 공정 소요시간 변수를 생성하는 함수를 작성합니다.

전체 공정 소요시간은 마지막 공정 완료시간 20_end_time 과 첫번째 공정 완료시간 04_end_time 의 차이(초)를 계산한 값입니다.

# 전체 및 개별 공정 소요시간 변수를 생성하는 함수입니다.
def gen_duration_feats(df, lst_stepsgap):
    
    # 전체 공정 소요시간(초) 변수를 생성합니다. 
    df['gen_tmdiff'] = (df['20_end_time'] - df['04_end_time']).dt.total_seconds()
    
    # 개별 스텝간 공정 소요시간(초) 변수를 생성합니다. 
    # ex. gen_tmdiff_0406 : 04 스텝 공정 완료 시간과 06 스텝 공정 완료 시간의 차이 
    for stepgap in lst_stepsgap:
        df[f'gen_tmdiff_{stepgap}'] = (df[f'{stepgap[2:]}_end_time'] - df[f'{stepgap[:2]}_end_time']).dt.total_seconds()

    return df

df_eda = gen_duration_feats(df_eda, lst_stepsgap)

Chamber 별 총 공정진행 시간에 따른 Target 값 변화

fig = plt.figure(figsize = (20, 40))
for i,name in enumerate(eda_df['module_name'].unique().tolist()):
    ax = plt.subplot(10, 5, i+1)
    tmp = df_eda[df_eda['module_name']==name]
    tmp['weekend'] = tmp['04_end_time'].dt.weekday.isin([5,6]).astype(int)
    tmp['date'] = tmp['04_end_time'].dt.date
    tmp = tmp.groupby(['module_name','date','weekend'])[['y','gen_tmdiff']].mean().reset_index()
    sns.scatterplot(data = tmp, x='gen_tmdiff', y='y', hue= 'weekend')
    plt.title(f'{name}', color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(loc='best',fontsize='xx-small')
    plt.subplots_adjust(hspace=0.5)
    
너무 총 공정 진행시간이 확 바뀐애들은 세정을 진행했거나 뭔가 APC를 바꿨을 듯.
세정을 하고 나아졌나 싶었는데, 다시 총 공정 진행시간이 빨라진것으로 보아, 그건 아닌듯싶다. 뭔가 공정 변경점이 있었던듯하다. EQ8은 공통적으로 10/14일때 늦어졌다가 다시 20일에 빨라짐. step 4-6-12 의 시간을 늘렸다. 10/14,17일 두번 계측에 느려진것
EQ7도 동일하게ㅣ 그때 느려졌다. 같은 스텝에 시간이 늘어났음.
챔버별로 보면 12-20 스텝은 거의 차이가 안나기도한다. 그럼 삭제하고 예측하는게 나을수도? 1초 차이가 나긴하네,,
그닥 큰 차이를 미칠거같진않지만 일단 진행. 나중에 FEATURE SELECTION에서 고려하자.

# 전체 공정 소요시간 및 개별 공정 소요시간 컬럼을 리스트 형태로 추출합니다. 
col_tmdiff = df_eda.filter(regex='gen_tmdiff($|_\d)').columns.tolist()

# 공정 소요시간의 통계값(최소, 최대, 평균)을 추출하는 함수입니다.
def tmdiff_stats(x):
    return [x.min(), x.max(), x.mean(), x.median()]

df_tmp = df_eda[col_tmdiff].apply(tmdiff_stats).T
df_tmp.columns = ['MIN', 'MAX', 'MEAN','MEDIAN']
df_tmp

# 전체 공정 소요시간 및 개별 공정 소요시간 컬럼을 리스트 형태로 추출합니다. 
col_tmdiff = df_eda.filter(regex='gen_tmdiff($|_\d)').columns.tolist()

# 공정 소요시간의 통계값(최소, 최대, 평균)을 추출하는 함수입니다.
def tmdiff_stats(x):
    return [x.min(), x.max(), x.mean(), x.median()]

df_tmp = df_eda[col_tmdiff].apply(tmdiff_stats).T
df_tmp.columns = ['MIN', 'MAX', 'MEAN','MEDIAN']
df_tmp

# 시각화를 위하여 1870초를 기준으로 일찍 마친 장비와 늦께 마친 장비를 구분합니다.
df_eda.loc[df_eda['gen_tmdiff'] < 1870, 'tmdiff_speed'] = 'E' # Early 
df_eda.loc[df_eda['gen_tmdiff'] > 1870, 'tmdiff_speed'] = 'L' # Late

fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10,5))
sns.scatterplot(x='gen_tmdiff', y='y', hue='tmdiff_speed', data = df_eda, ax=axes[0])
sns.scatterplot(x='gen_tmdiff', y='y', hue='module_name_eq', data = df_eda, ax=axes[1])
axes[0].legend(loc='lower left', ncol=2, title='tmdiff_speed')
axes[1].legend(loc='lower left', ncol=1, title='module_name_eq', bbox_to_anchor=(1.04, 0))

전체 공정 소요시간은 30분대 공정이 완료된 장비와 31분대에 공정이 완료된 장비로 뚜렷이 구분됩니다.

대부분의 장비는 31분대에 완료되었으며, 일찍 공정을 마친 장비는 주로 EQ7, EQ8 모듈입니다.

4. 데이터 전처리

# 전처리를 위한 학습용 데이터와 평가용 데이터를 복사합니다.
df_train = train.copy()
df_predict = predict.copy()

# 모듈 이름의 상위 집계 단위를 나타내는 1개의 범주형 변수를 생성합니다(3.3절)
df_train = gen_cate_feats(df_train)
df_predict = gen_cate_feats(df_predict)
df_train.filter(regex='module_name_eq').head(2)

# 전체 및 개별 공정 소요시간 7개의 변수를 생성합니다(3.4절)
df_train = gen_duration_feats(df_train, lst_stepsgap)
df_predict = gen_duration_feats(df_predict, lst_stepsgap)
df_train.filter(regex='tmdiff').head(2)

분산이 0인 변수 제거
분산 기준 설정(Variance Thresholding)은 가장 기본적인 특성 선택 방법 중 하나입니다.
분산이 0 인 특징은 정보가 없으므로 해당 특징을 삭제합니다.

drop_cols = df_train.loc[:,df_train.nunique()==1].columns.tolist()
df_train = df_train.loc[:, df_train.nunique()!=1]
print(f'** {len(drop_cols)} Features to Drop by Low Variance')
print(f'{drop_cols}')

표준화
ROBUST SCALING 진행. -> NAN 뜨는거 해결하자. Q3=Q1이 돼서 그런거같은데,
''' CATEGORY 변수 처리 및 NUM FEATURE 정의 '''
module2idx = {}
for i, module in enumerate(df_train['module_name'].unique()):
    module2idx[module] = i
    
eq2idx = {}
for i, eq in enumerate(df_train['module_name_eq'].unique()):
    eq2idx[eq] = i
    
df_train['module_name'] = df_train['module_name'].apply(lambda x: module2idx[x])
df_train['module_name_eq'] = df_train['module_name_eq'].apply(lambda x: eq2idx[x])
df_train['module_name'] = df_train['module_name'].astype('category')
df_train['module_name_eq'] = df_train['module_name_eq'].astype('category')
    
df_predict['module_name'] = df_predict['module_name'].apply(lambda x: module2idx[x])
df_predict['module_name_eq'] = df_predict['module_name_eq'].apply(lambda x: eq2idx[x])
df_predict['module_name'] = df_predict['module_name'].astype('category')
df_predict['module_name_eq'] = df_predict['module_name_eq'].astype('category')    
    
# num feature 정의(y 제외하고 해야함.)
cat_features = ['module_name','module_name_eq']
num_features = list(df_train.columns[df_train.dtypes==float])
num_features.remove('y')

COLS = cat_features+num_features

# SCALING
medians = []
percentile_25 = []
percentile_75 = []

med = df_train.loc[:, num_features].median(axis=0)
q1 = np.percentile(df_train.loc[:, num_features],25,axis=0)
q3 = np.percentile(df_train.loc[:, num_features],75,axis=0)

medians.append(med)
percentile_25.append(q1)
percentile_75.append(q3)

df_train.loc[:,num_features] = (df_train.loc[:,num_features] - med)/(q3 - q1)


4.6 변수 선택 - 나중에
변수 선택은 주어진 데이터의 변수 중에서 모델링의 가장 적절한 변수만 선택하는 과정입니다.

변수 선택 방법은 필터 방법(Filter method), 래퍼 방법(Wrapper method), 임베디드 방법(Embeded method)이 있습니다.

필터 방법 : 통계량(ex. 상관계수, 카이제곱, 상호정보량)을 구하여 가장 뛰어난 특성을 선택하는 기법 (ex. SelectKBest)
래퍼 방법 : 시행착오를 통해 가장 높은 품질의 예측을 만드는 특성의 부분조합을 찾는 기법 (ex. RFE, RFECV)
임베디드 방법 : 결정트리 모델로부터 생성된 특징 중요도를 이용하여 특성을 선택하는 기법 (ex. SelectFromModel)
다음은 필터 방법중 상호정보량을 사용하여 중요 특성을 추출하는 예제코드입니다.

참가자들은 자유롭게 분석하며, 변수 선택 중 한가지 방법으로 참고할 수 있습니다.


모델링
from catboost import CatBoostRegressor
import xgboost as xgb
import smote_variants as sv
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
# import library
from imblearn.under_sampling import TomekLinks

import optuna
from optuna.samplers import TPESampler


# XGB REGRESSOR
# mean_squared_error 의 매개변수 squared 가 False 이면 RMSE 를 반환합니다.
def rmse(y_true, y_pred):
    return round(mean_squared_error(y_true, y_pred, squared=False), 4)

def objective(trial):
    params_xgb = {
        'optimizer':trial.suggest_categorical('optimizer',['gbtree','gblinear','dart']),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.01,0.012,0.014,0.016,0.018, 0.02]),
        'n_estimators': 10000,
        'max_depth': trial.suggest_categorical('max_depth', [5,7,9,11,13,15,17]),
        'random_state': trial.suggest_categorical('random_state', [2022]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.9, step=0.1 ),
        'enable_categorical': True
    }
    
    xgb.DMatrix(X, enable_categorical=True)
    # 학습 데이터 중 일부를 검증 데이터 셋으로 분할합니다. 
    X_train, X_valid, y_train, y_valid = train_test_split(X, y.values, test_size=0.1, random_state=71)

    model = xgb.XGBRegressor(**params_xgb)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=35,
        verbose=False
    )

    xgb_pred = model.predict(X_valid)
    rmse_val = rmse(y_valid, xgb_pred)
    
    return rmse_val

sampler = TPESampler(seed=42)
study = optuna.create_study(
    study_name="xgb_parameter_opt",
    direction="minimize",
    sampler=sampler,
)
study.optimize(objective, n_trials=30)
print("Best Score:", study.best_value)
print("Best trial:", study.best_trial.params)

# CATBOOST REGRESSOR
''' 평일용 '''
def objective_CAT(trial):
    param = {
      "random_state":42,
      'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.1),
      'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
      "n_estimators":trial.suggest_int("n_estimators", 100, 10000),
      "max_depth":trial.suggest_int("max_depth", 4, 11),
      'random_strength' :trial.suggest_int('random_strength', 0, 30),
      "colsample_bylevel":trial.suggest_float("colsample_bylevel", 0.4, 1.0),
      "l2_leaf_reg":trial.suggest_float("l2_leaf_reg",1e-8,3e-5),
      "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
      "max_bin": trial.suggest_int("max_bin", 200, 400),
      'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
      'boosting_type':trial.suggest_categorical('boosting_type', ['Plain', 'Ordered'])
  }
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=71)
  
    cat = CatBoostRegressor(loss_function='RMSE',**param, eval_metric='RMSE')
    cat.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid,y_valid)],
            early_stopping_rounds=35,cat_features=cat_features,
            verbose=100)
    cat_pred = cat.predict(X_valid)
    rmse_val = rmse(y_valid, cat_pred)
    
    return rmse_val
    
sampler = TPESampler(seed=42)
study = optuna.create_study(
    study_name="cat_parameter_opt",
    direction="minimize",
    sampler=sampler,
)
study.optimize(objective_CAT, n_trials=10)
print("Best Score:", study.best_value)
print("Best trial:", study.best_trial.params)
# 6.36 -> 9.4 됨.

cat = CatBoostRegressor(**study.best_params)
cat.fit(X,y,cat_features=cat_features, early_stopping_rounds=35, verbose=100)

with open('CAT_EVAL6.9647.p', 'wb') as f:
    pickle.dump(cat, f)

# Feature Selection

from probatus.feature_elimination import EarlyStoppingShapRFECV

# Run feature elimination
shap_elimination = EarlyStoppingShapRFECV(
    clf=cat, step=0.2, cv=10, scoring='neg_mean_squared_error', early_stopping_rounds=15, n_jobs=-1, eval_metric='rmse')
report = shap_elimination.fit_compute(X, y)

# Make plots
performance_plot = shap_elimination.plot()

# Get final feature set
final_features_set = shap_elimination.get_reduced_features_set(num_features=)

# 예측 결과 제출
predict['msure_val'] = np.exp(cat.predict(df_predict[COLS]))
df_submission = predict[['key_val', 'msure_val']] 
df_submission.head()

# 예측값에 결측치가 포함되어 있는지 확인합니다.
df_submission.isnull().sum()

# 예측값의 갯수가 평가용 데이터의 갯수와 동일한지 확인합니다.
assert len(df_submission) == len(predict)
print(f'No. of Predict DataSet : {len(predict)}\nNo. of Submission DataSet : {len(df_submission)}')

# 예측 파일을 저장합니다. 
# 제출용 파일 이름은 cds_submission_팀명_차수.csv 형태로 제출합니다.
df_submission.set_index('key_val', inplace=True)
df_submission.to_csv('cds_submission_MyTeam_4.csv')


cv 하는거 참조
https://dacon.io/en/competitions/official/235877/codeshare/4710


**************************************************************22.07.06*******************************************************

ANN
from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow.keras.utils import to_categorical
from keras.layers import BatchNormalization, Activation
from keras.models import Sequential
from keras.layers.core import Dense

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers
from keras import backend as K

def identity(arg):
    return arg

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
        
        
def create_model():
    model = Sequential()
    model.add(Dense(128, input_dim=240))
    model.add(BatchNormalization())                    # Add Batchnorm layer before Activation
    model.add(Activation('relu'))  
    model.add(Dense(64))
    model.add(BatchNormalization())                    # Add Batchnorm layer before Activation
    model.add(Activation('relu'))  
    model.add(Dense(32))
    model.add(BatchNormalization())                    # Add Batchnorm layer before Activation
    model.add(Activation('relu'))  
    model.add(Dense(1))
    model.compile(optimizer='adam',
                  loss=root_mean_squared_error,
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model
    
model = create_model()
model.fit(X_one_hot, y, epochs=100, batch_size=32, validation_split=0.15)


STEP별 PCA

df_final = df_train.copy()
df_predict_final = df_predict.copy()

step_lst = ['04','06','12','13','17','18','20']

def pca_pre(datas,pred_datas, cols, n, step):
    pca = decomposition.PCA(n_components = n)
    pca_array = pca.fit_transform(datas[cols])
    pca_array_pred = pca.transform(pred_datas[cols])
    pca_df_train = pd.DataFrame(data = pca_array, columns = ['{0}_pca{1}'.format(step, num) for num in range(n)])
    pca_df_pred = pd.DataFrame(data = pca_array_pred, columns = ['{0}_pca{1}'.format(step, num) for num in range(n)])
    return pca_df_train, pca_df_pred
    
for i,col in enumerate(step_lst):
    cols = df_final.filter(regex='^'+col).columns.tolist()
    cols = list(set(cols))
    cols.remove(col+'_end_time')
    n_cols = len(cols)
    pca = decomposition.PCA(n_components = n_cols-1)
    pca_array = pca.fit_transform(df_final[cols])

    result = pd.DataFrame({'설명가능한 분산 비율(고윳값)':pca.explained_variance_,\
             '기여율':pca.explained_variance_ratio_},\
            index=np.array([f"pca{num+1}" for num in range(n_cols-1)]))
    result['누적기여율'] = result['기여율'].cumsum()
    if len(result.loc[result['누적기여율']>=0.8,:].index) >=1:
        n = result.loc[result['누적기여율']>=0.8,:].index[0][-2]
        try:
            n = int(result.loc[result['누적기여율']>=0.8,:].index[0][-2:])
            df, df_p = pca_pre(df_final, df_predict_final, cols, n, col)
            df_final = pd.concat([df_final, df],axis=1)
            df_predict_final = pd.concat([df_predict_final, df_p],axis=1)
            df_final.drop(cols, axis=1, inplace=True)
            df_predict_final.drop(cols, axis=1, inplace=True)
        except ValueError:
            n = int(result.loc[result['누적기여율']>=0.8,:].index[0][-1])
            df, df_p = pca_pre(df_final, df_predict_final, cols, n, col)
            df_final = pd.concat([df_final, df],axis=1)
            df_predict_final = pd.concat([df_predict_final, df_p],axis=1)
            df_final.drop(cols, axis=1, inplace=True)
            df_predict_final.drop(cols, axis=1, inplace=True)
    else:
        print(cols)
        # 얘네는 그럼 줄일 수 없다? 혹은 drop이다.


ANN

he = keras.initializers.he_normal(seed=0)
def create_model():
    model = Sequential()
    model.add(Dense(128, input_dim=311, kernel_initializer=he))
    model.add(Activation('relu'))  
    model.add(Dense(64, kernel_initializer=he))
    model.add(Activation('relu'))   
    model.add(Dense(32, kernel_initializer=he))
                      # Add Batchnorm layer before Activation
    model.add(Activation('relu'))   
    model.add(BatchNormalization())  
    model.add(Dense(1))
    model.compile(optimizer='adam',
                  loss=root_mean_squared_error,
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model
    
model = create_model()
model.fit(X_one_hot, y, epochs=1000, batch_size=13, validation_split=0.15)
ann_pred = model.predict(X_predict_one_hot)



220708 회사에서

from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow.keras.utils import to_categorical
from keras.layers import BatchNormalization, Activation
from keras.models import Sequential
from keras.layers.core import Dense

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers
from keras import backend as K

def identity(arg):
    return arg

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
he = keras.initializers.he_normal(seed=0)

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

def RMSE(y, y_pred):
    rmse = mean_squared_error(y, y_pred) ** 0.5
    return rmse

def create_model(): 
    model = Sequential() 
    model.add(Dense(128, kernel_initializer=he)) 
    model.add(Activation('relu'))
    model.add(Dense(64, kernel_initializer=he)) 
    model.add(Activation('relu'))
    model.add(Dense(32, kernel_initializer=he)) # Add Batchnorm layer before Activation model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(1)) 
    model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]) 
    return model

# model = create_model() 
# model.fit(X_one_hot, y, epochs=1000, batch_size=13, validation_split=0.15) 
# ann_pred = model.predict(X_predict_one_hot)

es = EarlyStopping(monitor='val_rmse', mode='min', verbose=1, patience=35)
mc = ModelCheckpoint('best_model.h5', monitor='val_rmse', mode='min', save_best_only=True)

anns = [create_model() for i in range(47)]
idexss = 0
for i in range(len(train_idxs)):
    if i==0:
        train_idx = range(train_idxs[i])
        idexss += train_idxs[i]
    else:
        train_idx = range(train_idxs[i-1], idexss+train_idxs[i])
        idexss += train_idxs[i]
    X = df_final.loc[train_idx, num_features]
    y = df_final.loc[train_idx, 'y']
    
    
    X_cat = X
    y_cat = pd.Series(y.values)
    loo = LeaveOneOut()
    ann = anns[i]
    loo.get_n_splits(X_cat)
    rmsle = []
    for train_idx, test_idx in loo.split(X_cat):
        ann.fit(X_cat.iloc[train_idx,:], y_cat.iloc[train_idx],epochs=1200, batch_size=8, validation_split=0.15, callbacks=[es])
        ann_pred = ann.predict(X_cat.iloc[test_idx,:])
        rmsle_val = RMSE(y_cat.iloc[test_idx], ann_pred)
        rmsle.append(rmsle_val)
    print(np.mean(rmsle))
    
    ann.fit(X_cat, y_cat,epochs=1200, batch_size=8, validation_split=0.15, callbacks=[es,mc])
    anns[i] = ann
    print('{}번째 모델 훈련이 완료되었습니다.'.format(i+1))
    
    
