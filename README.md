# 학습용 데이터 
train = make_dataset(train_sensor, train_quality)
# 평가용 데이터 
predict = make_dataset(predict_sensor)

# 전체 및 개별 공정 소요시간 변수를 생성하는 함수입니다.
def gen_duration_feats(df, lst_stepsgap):
    
    # 전체 공정 소요시간(초) 변수를 생성합니다. 
    df['gen_tmdiff'] = (df['20_end_time'] - df['04_end_time']).dt.total_seconds()
    
    # 개별 스텝간 공정 소요시간(초) 변수를 생성합니다. 
    # ex. gen_tmdiff_0406 : 04 스텝 공정 완료 시간과 06 스텝 공정 완료 시간의 차이 
    
    for stepgap in lst_stepsgap:
        df[f'gen_tmdiff_{stepgap}'] = (df[f'{stepgap[2:]}_end_time'] - df[f'{stepgap[:2]}_end_time']).dt.total_seconds()

    return df
    
    
4. 데이터 전처리

# 전처리를 위한 학습용 데이터와 평가용 데이터를 복사합니다.
df_train = train.copy()
df_predict = predict.copy()
del train

# -----------------------------------
# 3 장 EDA 분석에 필요한 변수를 선언합니다.
# -----------------------------------

# 센서 컬럼과 날짜 컬럼을 정의합니다. 
col_sensor = df_train.iloc[:, 4:-7].columns.tolist() 
col_time = df_train.filter(regex='end').columns.tolist() 

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
df_train[col_time] = df_train[col_time].apply(pd.to_datetime)

''' 여기서부터가 중요 '''
for_col_filter = []
for step_para in lst_sensors:
    for para in step_para:
        para = para.split('_')[0]+'_'+para.split('_')[1]
        for_col_filter.append(para)
for_col_filter = sorted(list(set(for_col_filter)))

# 전체 및 개별 공정 소요시간 7개의 변수를 생성합니다(3.4절)
df_train = gen_duration_feats(df_train, lst_stepsgap)
df_predict = gen_duration_feats(df_predict, lst_stepsgap)
df_train.filter(regex='tmdiff').head(2)

''' Cyclic Transformation 적용 '''
def cyclic_transformation(df, cols):
    for col in cols:
        step = col[:2]
        df[col] = pd.to_datetime(df[col])
        df[step+'_'+'hour'] = df[col].dt.hour
        df[step+'_'+'month'] = df[col].dt.month
        df[step+'_'+'day'] = df[col].dt.day
        df[step+'_'+'weekday'] = df[col].dt.weekday
        
        ## cyclic transformation on hour
        df[step+'_'+'hour_sin'] = np.sin(2 * np.pi * df[step+'_'+'hour']/23.0)
        df[step+'_'+'hour_cos'] = np.cos(2 * np.pi * df[step+'_'+'hour']/23.0)
        ## cyclic transformation on date 
        df[step+'_'+'date_sin'] = -np.sin(2 * np.pi * (df[step+'_'+'month']+df[step+'_'+'day']/31)/12)
        df[step+'_'+'date_cos'] = -np.cos(2 * np.pi * (df[step+'_'+'month']+df[step+'_'+'day']/31)/12)
        ## cyclic transformation on month
        df[step+'_'+'month_sin'] = -np.sin(2 * np.pi * df[step+'_'+'month']/12.0)
        df[step+'_'+'month_cos'] = -np.cos(2 * np.pi * df[step+'_'+'month']/12.0)
        ## cyclic transformation on weekday
        df[step+'_'+'weekday_sin'] = -np.sin(2 * np.pi * (df[step+'_'+'weekday']+1)/7.0)
        df[step+'_'+'weekday_cos'] = -np.cos(2 * np.pi * (df[step+'_'+'weekday']+1)/7.0)
        
        df.drop(step+'_'+'month',axis=1,inplace=True)
        df.drop(step+'_'+'month_sin',axis=1,inplace=True)
        df.drop(step+'_'+'month_cos',axis=1,inplace=True)
        
        
endtime_col = df_train.filter(regex='end_time$').columns.tolist()
cyclic_transformation(df_train, endtime_col)
cyclic_transformation(df_predict, endtime_col)

[[ Category 변수 처리 ]]
''' CATEGORY 변수 처리 및 NUM FEATURE 정의 '''
module2idx = {}
for i, module in enumerate(df_train['module_name'].unique()):
    module2idx[module] = i
    
# eq2idx = {}
# for i, eq in enumerate(df_train['module_name_eq'].unique()):
#     eq2idx[eq] = i
    
def col2cat(df, col, dict):
    df[col] = df[col].apply(lambda x: dict[x])
    df[col] = df[col].astype('category')
    return df[col]

# module_name cat 화
col2cat(df_train, 'module_name', module2idx)
col2cat(df_predict, 'module_name', module2idx)
# eq cat 화
# col2cat(df_train, 'module_name_eq', eq2idx)
# col2cat(df_predict, 'module_name_eq', eq2idx)

[[ 각 챔버별 전처리 ]]
df_final = df_train.copy()
df_predict_final = df_predict.copy()

module_unique = df_final['module_name'].unique()
df_trains = [df_final[df_final['module_name']==eq] for eq in module_unique]
num_features_lst = []
df_predicts = [df_predict_final[df_predict_final['module_name']==eq] for eq in module_unique]
for i, (trains,predicts) in enumerate(zip(df_trains,df_predicts)):
    drop_col = []
    for para in for_col_filter:
        col = trains.filter(regex='^'+para).columns.tolist()
        duplicate_deleted_df = trains[col].T.drop_duplicates(subset=trains[col].T.columns, keep='first').T
        if len(trains[col].columns.difference(duplicate_deleted_df.columns))==0:  # 다른게 없으면 무시,
            continue
        else:
            drop_col.extend(trains[col].columns.difference(duplicate_deleted_df.columns).tolist())
    df_trains[i] = trains.drop(drop_col,axis=1)
    df_predicts[i] = predicts.drop(drop_col, axis=1)
    
    ''' feature 정의'''
    num_features = list(df_trains[i].columns[df_trains[i].dtypes==float])
    num_features.remove('y')
    date_features = list(df_trains[i].columns[df_trains[i].dtypes==np.int64])
    tmdiff_features = df_trains[i].filter(regex='^gen_').columns.tolist()
    col_numerical = num_features + date_features + tmdiff_features
    
    ''' 분산 0인 col 제거 '''
    thresholder = VarianceThreshold(threshold=0)
    _ = thresholder.fit_transform(df_trains[i][col_numerical])

     # 분산이 0이면 True 이므로 제거할 컬럼을 추출합니다.  
    mask = ~thresholder.get_support()
    cols_var_drop = np.asarray(col_numerical)[mask].tolist()
    print(f'** {len(cols_var_drop)} Features to Drop by Low Variance')
    print(f'{cols_var_drop}')
                                      
    df_trains[i] = trains.drop(cols_var_drop+['module_name'],axis=1)
    df_predicts[i] = predicts.drop(cols_var_drop+['module_name'], axis=1)
    
    ''' Cyclic Transformation 된 time만 사용. '''
    num_features = list(df_trains[i].columns[df_trains[i].dtypes==float])
    num_features.remove('y')
    num_features_lst.append(num_features)
    
모델링 진행

xgbs = []
for (train, num_f) in zip(df_trains,num_features_lst):
    def objective(trial):
        params_xgb = {
            'booster':trial.suggest_categorical('booster',['gbtree','dart']),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0),
            'colsample_bytree': trial.suggest_int('colsample_bytree', 0.3,1.0),
            'subsample': trial.suggest_float('subsample', 0.3, 1.0),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
            'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
            'max_depth': trial.suggest_int("max_depth", 4, 12),
            'random_state': trial.suggest_categorical('random_state', [0]),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
    #         'tree_method':'gpu_hist',
    #         'gpu_id':'0'
        }
        X = train[num_f]
        y = np.log1p(train['y'])

        model = xgb.XGBRegressor(**params_xgb)
        loo = LeaveOneOut()
        scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
        scores = np.sqrt(-scores)
        print('CV scores for {0}: {1}'.format([i,scores]))
        print('Mean score : ', np.mean(scores))
        rmsle_val = np.mean(scores)
     
        return rmsle_val
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
    study_name="xgb_parameter_opt",
    direction="minimize",
    sampler=sampler,
    )
    study.optimize(objective, n_trials=30)
    print("Best Score:", study.best_value)
    print("Best trial:", study.best_trial.params)
    
    model = xgb.XGBRegressor(**study.best_params)
    model.fit(train[num_f], np.log1p(train['y']))
    xgbs.append(model)
    
    
    
    ''' E와 L 나누려면 '''
df_final = df_train.copy()
df_predict_final = df_predict.copy()

E = df_final[df_final['gen_tmdiff']<=1870].copy()
L = df_final[df_final['gen_tmdiff']>1870].copy()

E_pred = df_predict_final[df_predict_final['gen_tmdiff']<=1870].copy()
L_pred = df_predict_final[df_predict_final['gen_tmdiff']>1870].copy()

''' E 중복열 제거 '''
drop_col = []
for para in for_col_filter:
    col = E.filter(regex='^'+para).columns.tolist()
    duplicate_deleted_df = E[col].T.drop_duplicates(subset=E[col].T.columns, keep='first').T
    if len(E[col].columns.difference(duplicate_deleted_df.columns))==0:  # 다른게 없으면 무시,
        continue
    else:
        drop_col.extend(E[col].columns.difference(duplicate_deleted_df.columns).tolist())
E = E.drop(drop_col,axis=1)
E_pred = E_pred.drop(drop_col, axis=1)
print('E에서 중복열 :', drop_col)

''' L 중복열 제거 '''

drop_col = []
for para in for_col_filter:
    col = L.filter(regex='^'+para).columns.tolist()
    duplicate_deleted_df = L[col].T.drop_duplicates(subset=L[col].T.columns, keep='first').T
    if len(L[col].columns.difference(duplicate_deleted_df.columns))==0:  # 다른게 없으면 무시,
        continue
    else:
        drop_col.extend(L[col].columns.difference(duplicate_deleted_df.columns).tolist())
L = L.drop(drop_col,axis=1)
L_pred = L_pred.drop(drop_col, axis=1)
print('L에서 중복열 :', drop_col)

''' feature 정의'''
num_features = list(E.columns[E.dtypes==float])
num_features.remove('y')
# date_features = list(E.columns[E.dtypes==np.int64])
col_numerical = num_features
    
''' 분산 0인 col 제거 '''
thresholder = VarianceThreshold(threshold=0)
_ = thresholder.fit_transform(E[col_numerical])

 # 분산이 0이면 True 이므로 제거할 컬럼을 추출합니다.  
mask = ~thresholder.get_support()
cols_var_drop = np.asarray(col_numerical)[mask].tolist()
print(f'** {len(cols_var_drop)} Features to Drop by Low Variance')
print(f'{cols_var_drop}')
E = E.drop(cols_var_drop,axis=1)

''' feature 정의'''
num_features = list(L.columns[L.dtypes==float])
num_features.remove('y')
# date_features = list(L.columns[L.dtypes==np.int64])
col_numerical = num_features
    
''' 분산 0인 col 제거 '''
thresholder = VarianceThreshold(threshold=0)
_ = thresholder.fit_transform(L[col_numerical])

 # 분산이 0이면 True 이므로 제거할 컬럼을 추출합니다.  
mask = ~thresholder.get_support()
cols_var_drop = np.asarray(col_numerical)[mask].tolist()
print(f'** {len(cols_var_drop)} Features to Drop by Low Variance')
print(f'{cols_var_drop}')
L = L.drop(cols_var_drop,axis=1)

''' feature 재정의'''
E_num_features = list(E.columns[E.dtypes==float])
cat_features = ['module_name']
# date_features = list(df_final.columns[df_final.dtypes==np.int64])
E_train = E_num_features + cat_features
E_num_features.remove('y')
E_predict = E_num_features + cat_features

''' feature 재정의'''
L_num_features = list(L.columns[L.dtypes==float])
# date_features = list(df_final.columns[df_final.dtypes==np.int64])
L_train = L_num_features + cat_features
L_num_features.remove('y')
L_predict = L_num_features + cat_features

def prep_cate_feats(df_tr, df_te, feat_nm):

    df_merge = pd.concat([df_tr, df_te])

    # 컬럼명과 범주형 변수의 레벨명을 이용한 새로운 컬럼명을 자동생성합니다. 
    # ex. module_name_eq -> module_name_eq_EQ01, module_name_eq_EQ02, etc.
    df_merge = pd.get_dummies(df_merge, columns=[feat_nm])

    df_tr = df_merge.iloc[:df_tr.shape[0], :].reset_index(drop=True)
    df_te = df_merge.iloc[df_tr.shape[0]:, :].reset_index(drop=True)

    return df_tr, df_te

# module_name_eq 의 원-핫 인코딩 변수를 생성합니다.
E, E_pred = prep_cate_feats(E[E_train], E_pred[E_predict], 'module_name')
L, L_pred = prep_cate_feats(L[L_train], L_pred[L_predict], 'module_name')

''' feature 재정의'''
E_num_features = list(E.columns[E.dtypes==float])
module_col = E.filter(regex='^module_name').columns.tolist()
E_num_features.remove('y')
# date_features = list(df_final.columns[df_final.dtypes==np.int64])
E_train = E_num_features + module_col
E_predict = E_num_features + module_col

''' feature 재정의'''
L_num_features = list(L.columns[L.dtypes==float])
L_num_features.remove('y')
# date_features = list(df_final.columns[df_final.dtypes==np.int64])
L_train = L_num_features + module_col
L_predict = L_num_features + module_col


''' 혹시 ngb '''
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from ngboost import NGBRegressor
from ngboost.learners import default_tree_learner
from ngboost.distns import Normal
from ngboost.scores import MLE, CRPS

def objective_NGB(trial):
    param = {
      "random_state":42,
      'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.5),
      "n_estimators":trial.suggest_int("n_estimators", 100, 10000),
      "col_sample":trial.suggest_float("colsample", 0.2, 1.0),
    'natural_gradient':trial.suggest_categorical("natural_gradient", [True,False]),
    'verbose_eval':trial.suggest_int("verbose_eval", 10, 80),  

  }
    X_train, X_valid, y_train, y_valid = train_test_split(X_one_hot, y, test_size=0.2, shuffle=True, random_state=71)
    ngb = NGBRegressor(Base=default_tree_learner, Dist=Normal,Score=MLE, **param)
    ngb.fit(X_train,y_train, X_val=X_valid, Y_val=y_valid, early_stopping_rounds=20)
    
    ngb_pred = ngb.predict(X_valid)
    rmsle_val = RMSE(y_valid, ngb_pred)

    return rmsle_val
    
sampler = TPESampler(seed=42)
study = optuna.create_study(
    study_name="xgb_parameter_opt",
    direction="minimize",
    sampler=sampler,
)
study.optimize(objective_NGB, n_trials=10)
print("Best Score:", study.best_value)
print("Best trial:", study.best_trial.params)
