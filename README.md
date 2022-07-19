''' 최신 '''

# 각 챔버별 전처리
df_final = df_train.copy()
df_predict_final = df_predict.copy()

module_unique = df_final['module_name'].unique()
df_trains = [df_final[df_final['module_name']==eq] for eq in module_unique]
num_features_lst = []
df_predicts = [df_predict_final[df_predict_final['module_name']==eq] for eq in module_unique]
''' 중복되는 열 제거하기. '''
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
    col_numerical = num_features + date_features
    
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
    
    ''' 1부터도 log transformation시 skewness와 kurtois가 많이 줄어듦. 효과적일 것이라 변환진행 ex) '20_time_para42' '''
    skew1_df = ((trains.skew()>1)|(trains.skew()<-1)).reset_index().iloc[1:].reset_index(drop=True)
    skew1_df.columns=['param','boolean']
    high_skew1_col = skew1_df.loc[skew1_df['boolean'],:]['param'].unique().tolist()
    ''' 04_fr_para28 , 06_fr_para28 음수를 가진 col인데, skew는 높지만 시각화시 괜찮아서 제외. '''
    minus_col = trains[high_skew1_col][trains[high_skew1_col]<=0].dropna(axis=1).columns.tolist()
    high_skew1_col = [x for x in high_skew1_col if x not in minus_col]
    print(len(high_skew1_col))

    trains[high_skew1_col] = np.log1p(trains[high_skew1_col])
    predicts[high_skew1_col] = np.log1p(predicts[high_skew1_col])
    df_trains[i] = trains
    df_predicts[i] = predicts
    
    ''' Cyclic Transformation 된 time만 사용. gen+float f들 '''
    num_features = list(trains.columns[trains.dtypes==float])
    num_features.remove('y')
    num_features_lst.append(num_features)
    
    
# 모델링

** LGB **

lgbs = []
lgb_scores = []
for i, (train, num_f) in enumerate(zip(df_trains, num_features_lst)):
    def objective_LGB(trial):
        param_lgb = {
            'objective':'regression',
            'metric':'rmse',
            "random_state":42,
            'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 4e-5),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 9e-2),
            'bagging_fraction' :trial.suggest_loguniform('bagging_fraction', 0.01, 1.0),
            "n_estimators":trial.suggest_int("n_estimators", 1000, 10000),
            "max_depth":trial.suggest_int("max_depth", 1, 20),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "max_bin": trial.suggest_int("max_bin", 200, 500)
        }
        X = train[num_f]
        y = np.log1p(train['y'])

        model = lgb.LGBMRegressor(**param_lgb)
        loo = LeaveOneOut()
        scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
        scores = np.sqrt(-scores)
        print(f'CV scores for {i}: {scores}')
        print('Mean score : ', np.mean(scores))
        rmsle_val = np.mean(scores)
     
        return rmsle_val
    
    sampler = TPESampler(seed=42)
    study_lgb = optuna.create_study(
                study_name="lgb_parameter_opt",
                direction="minimize",
                sampler=sampler,
            )
    study_lgb.optimize(objective_LGB, n_trials=5)
    print("Best Score:", study_lgb.best_value)
    print("Best trial:", study_lgb.best_trial.params)
    lgb_scores.append(study_lgb.best_value)
    
    model = lgb.LGBMRegressor(**study_lgb.best_params)
    model.fit(train[num_f], np.log1p(train['y']))
    print('{}th model training is completed'.format(i+1))
    lgbs.append(model)
    
    
** XGB **

xgbs = []
xgb_scores = []
for i, (train, num_f) in enumerate(zip(df_trains, num_features_lst)):
    def objective_XGB(trial):
        param = {
            'booster':'gblinear',
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
#             "reg_alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
#             'colsample_bytree': trial.suggest_int('colsample_bytree', 0.3, 1.0),
#             'subsample': trial.suggest_float('subsample', 0.3, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.8),
            'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
#             'max_depth': trial.suggest_int("max_depth", 4, 12),
            'random_state': 0,
#             'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
#             'tree_method':'exact'
#             'tree_method':'gpu_hist',
#             'gpu_id':'0'
        }
        
#         if param["booster"] in ["gbtree", "dart"]:
#         # maximum depth of the tree, signifies complexity of the tree.
#             param["max_depth"] = trial.suggest_int("max_depth", 3, 12, step=2)
#             # minimum child weight, larger the term more conservative the tree.
#             param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
#             param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
#             # defines how selective algorithm is.
#             param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
#             param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

#         if param["booster"] == "dart":
#             param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
#             param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
#             param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
#             param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
        
        X = train[num_f]
        y = np.log1p(train['y'])

        model = xgb.XGBRegressor(**param)
        cv = KFold(5,shuffle=True, random_state=0)
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        scores = np.sqrt(-scores)
        print(f'CV scores for {i}: {scores}')
        print('Mean score : ', np.mean(scores))
        rmsle_val = np.mean(scores)
     
        return rmsle_val
    
    sampler = TPESampler(seed=42)
    study_xgb = optuna.create_study(
            study_name="xgb_parameter_opt",
            direction="minimize",
            sampler=sampler,
    )
    study_xgb.optimize(objective_XGB, n_trials=5)
    print("Best Score:", study_xgb.best_value)
    print("Best trial:", study_xgb.best_trial.params)
    xgb_scores.append(study_xgb.best_value)
    
    model = xgb.XGBRegressor(**study_xgb.best_params)
    model.fit(train[num_f], np.log1p(train['y']))
    print('{}th model training is completed'.format(i+1))
    xgbs.append(model)


** CAT **

cats = []
cat_scores= []
for i, (train, num_f) in enumerate(zip(df_trains, num_features_lst)):
    def objective_CAT(trial):
        param = {
          "random_state":42,
          'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.5),
          'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
          "n_estimators":trial.suggest_int("n_estimators", 100, 10000),
          "max_depth":trial.suggest_int("max_depth", 4, 12),
          'random_strength' :trial.suggest_int('random_strength', 0, 30),
          "colsample_bylevel":trial.suggest_float("colsample_bylevel", 0.4, 1.0),
          "l2_leaf_reg":trial.suggest_float("l2_leaf_reg",0,10),
          "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
          "max_bin": trial.suggest_int("max_bin", 200, 400),
          'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
          'boosting_type':trial.suggest_categorical('boosting_type', ['Plain', 'Ordered']),
#           'task_type':'GPU',
#           'devices':'0:16',
#           'iterations':100,
#           'rsm':1
        }
        X = train[num_f]
        y = np.log1p(train['y'])

        model = CatBoostRegressor(**param,loss_function='RMSE', eval_metric='RMSE')
        loo = LeaveOneOut()
        scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
        scores = np.sqrt(-scores)
        print(f'CV scores for {i}: {scores}')
        print('Mean score : ', np.mean(scores))
        rmsle_val = np.mean(scores)

        return rmsle_val
    
    sampler = TPESampler(seed=42)
    study_cat = optuna.create_study(
            study_name="cat_parameter_opt",
            direction="minimize",
            sampler=sampler,
            )
    study_cat.optimize(objective_CAT, n_trials=5)
    print("Best Score:", study_cat.best_value)
    print("Best trial:", study_cat.best_trial.params)
    cat_scores.append(study_cat.best_value)
    
    model = CatBoostRegressor(**study_cat.best_params,
                                loss_function='RMSE', eval_metric='RMSE',
                                task_type='GPU',devices='0:16',rsm=1)
    model.fit(train[num_f], np.log1p(train['y']))
    print('{}th model training is completed'.format(i+1))
    cats.append(model)


** RIDGE **

ridges = []
ridge_scores = []
for i, (train, num_f) in enumerate(zip(df_trains, num_features_lst)):
    def objective_RIDGE(trial):
        param = {
          "random_state":42,
            'alpha':trial.suggest_float("alpha",0.1,10),
            'fit_intercept':trial.suggest_categorical('fit_intercept', [True, False]),
            'normalize':trial.suggest_categorical('normalize', [True, False]),
        }
        X = train[num_f]
        y = np.log1p(train['y'])

        model = Ridge(**param)
        loo = LeaveOneOut()
        
        scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
        scores = np.sqrt(-scores)
        print(f'CV scores for {i}: {scores}')
        print('Mean score : ', np.mean(scores))
        rmsle_val = np.mean(scores)

        return rmsle_val
    
    sampler = TPESampler(seed=42)
    study_ridge = optuna.create_study(
            study_name="ridge_parameter_opt",
            direction="minimize",
            sampler=sampler,
            )
    study_ridge.optimize(objective_RIDGE, n_trials=10)
    print("Best Score:", study_ridge.best_value)
    print("Best trial:", study_ridge.best_trial.params)
    ridge_scores.append(study_ridge.best_value)
    
    model = Ridge(**study_ridge.best_params)
    model.fit(train[num_f], np.log1p(train['y']))
    print('{} model training is completed'.format(i+1))
    ridges.append(model)


BAYESIANRIDGE

brs = []
br_scores = []
for i, (train, num_f) in enumerate(zip(df_trains, num_features_lst)):
    def objective_BR(trial):
        param = {
            'n_iter':trial.suggest_int("n_iter",10,500),
            'alpha_2':trial.suggest_uniform("alpha_2",-10,10),
            'lambda_2' :trial.suggest_uniform('lambda_2', -10, 10),
            'fit_intercept':trial.suggest_categorical('fit_intercept', [True, False]),
            'normalize':trial.suggest_categorical('normalize', [True, False]),
        }
        X = train[num_f]
        y = np.log1p(train['y'])

        model = BayesianRidge(**param)
        loo = LeaveOneOut()

        scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
        scores = np.sqrt(-scores)
        print(f'CV scores for {i}: {scores}')
        print('Mean score : ', np.mean(scores))
        rmsle_val = np.mean(scores)

        return rmsle_val
    
    sampler = TPESampler(seed=42)
    study_br = optuna.create_study(
            study_name="BayesianRidge_parameter_opt",
            direction="minimize",
            sampler=sampler,
            )
    study_br.optimize(objective_BR, n_trials=10)
    print("Best Score:", study_br.best_value)
    print("Best trial:", study_br.best_trial.params)
    br_scores.append()
    
    model = BayesianRidge(**study_br.best_params)
    model.fit(train[num_f], np.log1p(train['y']))
    print('{} model training is completed'.format(i+1))
    brs.append(model)
    
    
챔버별 최상의 CV SCORE 탐색 및 THRESHOLD 도출


score_df = pd.DataFrame({'model':['lgb']*60 + ['xgb']*60 + ['cat']*60 + ['ridge']*60+['br']*60,
                         'chamber': list(range(0,47))*5,
                         'RMSLE' : lgb_scores + xgb_scores + cat_scores + ridge_scores + br_scores})

fig = plt.figure(figsize = (10, 30))
sns.barplot(data = score_df, orient = 'h', x = 'RMSLE', y = 'chamber', hue = 'model')


























    
    
    
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


''' REGRESSION OVERSAMPLING 참고 '''
dacon.io/competitions/official/235877/codeshare/4711


'' 혹시 HUBERREGRESSOR ''
hbs = []
hb_scores = []
for i, (train, num_f) in enumerate(zip(df_trains, num_features_lst)):
    def objective_HB(trial):
        param = {
            'epsilon':trial.suggest_float("epsilon",1.0,5.0),
            'alpha':trial.suggest_float("alpha",0.0001,5)
        }
        X = train[num_f]
        y = np.log1p(train['y'])

        model = HuberRegressor(**param)
        loo = LeaveOneOut()

        scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
        scores = np.sqrt(-scores)
        print(f'CV scores for {i}: {scores}')
        print('Mean score : ', np.mean(scores))
        rmsle_val = np.mean(scores)

        return rmsle_val
    
    sampler = TPESampler(seed=42)
    study_hb = optuna.create_study(
            study_name="HB_parameter_opt",
            direction="minimize",
            sampler=sampler,
            )
    study_hb.optimize(objective_HB, n_trials=10)
    print("Best Score:", study_hb.best_value)
    print("Best trial:", study_hb.best_trial.params)
    hb_scores.append(study_hb.best_value)
    
    model = HuberRegressor(**study_hb.best_params)
    model.fit(train[num_f], np.log1p(train['y']))
    print('{} model training is completed'.format(i+1))
    hbs.append(model)
    
    

세로줄긋기
plt.axvline(0.005)


최상의 cv score columns 변경해야함.
# (건물 별 모델 cv score 의 pivot_q quantile 값*threshold) 보다 작은 cv score를 가진 모델만 건물별로 선택
def good_models(score_df, pivot_q, threshold):
    score_pivot = pd.DataFrame(score_df.pivot('chamber', 'model', 'RMSLE').values,
                               columns = ['br','cat','en','lgb', 'ridge'])
    li = []
    for i in range(len(score_pivot)):
        temp = score_pivot.iloc[i]
        q = temp.quantile(pivot_q)
        best = list(temp[temp <= threshold*q].index)
        li.append(best)
    return li
''' RandomForest '''
rfs = []
rf_scores = []
for i, (train, num_f) in enumerate(zip(df_trains, num_features_lst)):
    def objective_RF(trial):
        param = {
            #"device_type": trial.suggest_categorical("device_type", ['gpu']),
            "n_estimators": trial.suggest_int("n_estimators", 100,2000,step=10),
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 30),
            "random_state": trial.suggest_categorical("random_state", [2022]),
        }
        
        X = train[num_f]
        y = np.log1p(train['y'])

        model = RandomForestRegressor(**param)  
        loo = LeaveOneOut()
        scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
        scores = np.sqrt(-scores)
        print(f'CV scores for {i}: {scores}')
        print('Mean score : ', np.mean(scores))
        rmsle_val = np.mean(scores)
     
        return rmsle_val
    
    sampler = TPESampler(seed=42)
    study_rf = optuna.create_study(
                study_name="rf_parameter_opt",
                direction="minimize",
                sampler=sampler,
            )
    study_rf.optimize(objective_RF, n_trials=5)
    print("Best Score:", study_rf.best_value)
    print("Best trial:", study_rf.best_trial.params)
    rf_scores.append(study_rf.best_value)
    
    model = RandomForestRegressor(**study_rf.best_params)
    model.fit(train[num_f], np.log1p(train['y']))
    print('{}th model training is completed'.format(i+1))
    rfs.append(model)


''' scaling 진행한 data 준비 '''
module_unique = df_final['module_name'].unique()
df_trains = [df_final[df_final['module_name']==eq] for eq in module_unique]
df_predicts = [df_predict_final[df_predict_final['module_name']==eq] for eq in module_unique]
df_trains_scaled = []
df_predicts_scaled = []
# num_features_lst = []

''' 중복되는 열 제거하기. '''
for i, (trains,predicts) in enumerate(zip(df_trains,df_predicts)):
    drop_col = []
    for para in for_col_filter:
        col = trains.filter(regex='^'+para).columns.tolist()
        duplicate_deleted_df = trains[col].T.drop_duplicates(subset=trains[col].T.columns, keep='first').T
        if len(trains[col].columns.difference(duplicate_deleted_df.columns))==0:  # 다른게 없으면 무시,
            continue
        else:
            drop_col.extend(trains[col].columns.difference(duplicate_deleted_df.columns).tolist())
    trains.drop(drop_col,axis=1,inplace=True)
    predicts.drop(drop_col, axis=1, inplace=True)
    
    var0_cols = trains.loc[:,trains.nunique()==1].columns.tolist()
    print(f'module{i}의 drop할 columns : {var0_cols}')
    trains.drop(var0_cols, axis=1, inplace=True)
    predicts.drop(var0_cols, axis=1, inplace=True)
    
    ''' Cyclic Transformation 된 time만 사용. gen+float f들 '''
    num_features = list(trains.columns[trains.dtypes==float])
    num_features.remove('y')
    num_features_lst.append(num_features)
    
    scaler = StandardScaler()
    scaled_trains = scaler.fit_transform(trains[num_features])
    scaled_predicts = scaler.transform(predicts[num_features])
    df_trains_scaled.append(scaled_trains)
    df_predicts_scaled.append(scaled_predicts)
    
    df_trains[i] = trains
    df_predicts[i] = predicts
    
#     ''' Cyclic Transformation 된 time만 사용. gen+float f들 '''
#     num_features = list(trains.columns[trains.dtypes==float])
#     num_features.remove('y')
#     num_features_lst.append(num_features)

''' 오버샘플링 '''
from imblearn.under_sampling import RandomUnderSampler

    assert predicts[high_skew1_col].isnull().sum().sum() == 0
    assert (predicts[high_skew1_col]==float('-inf')).sum().sum() == 0
    assert (predicts[high_skew1_col]==float('inf')).sum().sum() == 0
    
    var0_cols = trains.loc[:,trains.nunique()==1].columns.tolist()
    print(f'module{i}의 drop할 columns : {var0_cols}')
    trains.drop(var0_cols, axis=1, inplace=True)
    predicts.drop(var0_cols, axis=1, inplace=True)
    
    df_trains[i] = trains
    df_predicts[i] = predicts
    
    ''' Cyclic Transformation 된 time만 사용. gen+float f들 '''
    num_features = list(trains.columns[trains.dtypes==float])
    num_features.remove('y')
    num_features_lst.append(num_features)
    
    sampler = RandomUnderSampler(random_state=42)
    X,y = trains = sampler.fit_resample(trains[num_features], trains['y'])
    
    이렇게 확인해야할듯 이후는  scaling 진행한 거와 동일하게 진행. 
    
    
    
    
    
    
    ''' 챔버별 모델링 '''
df_final = df_train.copy()
df_predict_final = df_predict.copy()

module_unique = df_final['module_name'].unique()
df_trains = [df_final[df_final['module_name']==eq] for eq in module_unique]
df_predicts = [df_predict_final[df_predict_final['module_name']==eq] for eq in module_unique]
df_trains_scaled = []
df_predicts_scaled = []
modeling_col_lst = []
targets = []

''' 중복되는 열 제거하기. '''
for i, (trains,predicts) in enumerate(zip(df_trains,df_predicts)):
    drop_col = []
    for para in for_col_filter:
        col = trains.filter(regex='^'+para).columns.tolist()
        if col:
            duplicate_deleted_df = trains[col].T.drop_duplicates(subset=trains[col].T.columns, keep='first').T
            if len(trains[col].columns.difference(duplicate_deleted_df.columns))==0:  # 다른게 없으면 무시,
                continue
            else:
                drop_col.extend(trains[col].columns.difference(duplicate_deleted_df.columns).tolist())
        else:
            continue
    trains.drop(drop_col,axis=1,inplace=True)
    predicts.drop(drop_col, axis=1, inplace=True)
    
    var0_cols = trains.loc[:,trains.nunique()==1].columns.tolist()
    print(f'module{i}의 drop할 columns : {var0_cols}')
    trains.drop(var0_cols, axis=1, inplace=True)
    predicts.drop(var0_cols, axis=1, inplace=True)
    
    ''' Cyclic Transformation 된 time만 사용. gen+float f들 '''
    num_features = list(trains.columns[trains.dtypes==float])
    targets.append(pd.Series(trains['y']))
    num_features.remove('y')
    test_date_features = trains.columns[trains.dtypes==np.int64].tolist()
    COLS = test_date_features + num_features
    modeling_col_lst.append(COLS)
    
    scaler = StandardScaler()
    trains.loc[:, num_features] = scaler.fit_transform(trains[num_features])
    predicts.loc[:, num_features] = scaler.transform(predicts[num_features])
    df_trains_scaled.append(trains)
    df_predicts_scaled.append(predicts)
    
    df_trains[i] = trains
    df_predicts[i] = predicts
    
#     ''' Cyclic Transformation 된 time만 사용. gen+float f들 '''
#     num_features = list(trains.columns[trains.dtypes==float])
#     num_features.remove('y')
#     num_features_lst.append(num_features)

lgbs = []
lgb_scores = []
for i, (train, cols, y)in enumerate(zip(df_trains, modeling_col_lst,targets)):
    def objective_LGB(trial):
        param_lgb = {
            'objective':'regression',
            'metric':'rmse',
            "random_state":42,
            'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 4e-5),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 9e-2),
            'bagging_fraction' :trial.suggest_loguniform('bagging_fraction', 0.01, 1.0),
            "n_estimators":trial.suggest_int("n_estimators", 100, 1000),
            "max_depth":trial.suggest_int("n_estimators", 3, 12),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "max_bin": trial.suggest_int("max_bin", 200, 500)
        }
        X_lgb = train[cols]
        y_lgb = np.log1p(y)

        model = lgb.LGBMRegressor(**param_lgb)
        loo = LeaveOneOut()
        scores = cross_val_score(model, X_lgb, y_lgb, cv=loo, scoring='neg_mean_squared_error')
        scores = np.sqrt(-scores)
        print(f'CV scores for {i}: {scores}')
        print('Mean score : ', np.mean(scores))
        rmsle_val = np.mean(scores)
     
        return rmsle_val
    
    sampler = TPESampler(seed=42)
    study_lgb = optuna.create_study(
                study_name="lgb_parameter_opt",
                direction="minimize",
                sampler=sampler,
            )
    study_lgb.optimize(objective_LGB, n_trials=3)
    print("Best Score:", study_lgb.best_value)
    print("Best trial:", study_lgb.best_trial.params)
    lgb_scores.append(study_lgb.best_value)
    
    model = lgb.LGBMRegressor(**study_lgb.best_params)
    model.fit(train[cols], np.log1p(y))
    print('{}th model training is completed'.format(i+1))
    lgbs.append(model)
    
xgbs = []
xgb_scores = []
for i, (train, cols, y)in enumerate(zip(df_trains, modeling_col_lst,targets)):
    def objective_XGB(trial):
        param = {
            'booster': 'gbtree',
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 9e-2),
#             "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
            'random_state': 0,
#             "tree_method": "gpu_hist",
#             'gpu_id':'0',
            # maximum depth of the tree, signifies complexity of the tree.
            "max_depth": trial.suggest_int("max_depth", 3, 12, step=2),
            # minimum child weight, larger the term more conservative the tree.
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
            "eta": trial.suggest_float("eta", 0.01, 0.8),
            # defines how selective algorithm is.
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            'colsample_bytree': trial.suggest_int('colsample_bytree', 0.3, 1.0)
        
    }
        
        
        X_xgb = train[cols]
        y_xgb = np.log1p(y)

        model = xgb.XGBRegressor(**param)
        cv = KFold(5,shuffle=True, random_state=0)
        scores = cross_val_score(model, X_xgb, y_xgb, cv=cv, scoring='neg_mean_squared_error')
        scores = np.sqrt(-scores)
        print(f'CV scores for {i}: {scores}')
        print('Mean score : ', np.mean(scores))
        rmsle_val = np.mean(scores)
     
        return rmsle_val
    
    sampler = TPESampler(seed=42)
    study_xgb = optuna.create_study(
            study_name="xgb_parameter_opt",
            direction="minimize",
            sampler=sampler,
    )
    study_xgb.optimize(objective_XGB, n_trials=5)
    print("Best Score:", study_xgb.best_value)
    print("Best trial:", study_xgb.best_trial.params)
    xgb_scores.append(study_xgb.best_value)
    
    model = xgb.XGBRegressor(**study_xgb.best_params)
    model.fit(train[cols], np.log1p(y))
    print('{}th model training is completed'.format(i+1))
    xgbs.append(model)
    
    
ens = []
en_scores = []
for i, (train, cols, y)in enumerate(zip(df_trains_scaled, modeling_col_lst, targets)):
    def objective_en(trial):
        param = {
            'alpha':trial.suggest_float("alpha",1e-2,20),
            'fit_intercept':trial.suggest_categorical('fit_intercept', [True, False]),
            'normalize':trial.suggest_categorical('normalize', [True, False]),
            'tol':trial.suggest_float("tol",1e-6,1e-2),
            'selection':trial.suggest_categorical('selection', ['cyclic','random']),
            'l1_ratio':trial.suggest_float("l1_ratio",1e-6,1.0)
        }
        X_en = train[cols]
        y_en = np.log1p(y)

        model = ElasticNet(**param, random_state=42)
        loo = LeaveOneOut()
        scores = cross_val_score(model, X_en, y_en, cv=loo, scoring='neg_mean_squared_error')
        scores = np.sqrt(-scores)
        print(f'CV scores for {i}: {scores}')
        print('Mean score : ', np.mean(scores))
        rmsle_val = np.mean(scores)

        return rmsle_val
    
    sampler = TPESampler(seed=42)
    study_en = optuna.create_study(
            study_name="en_parameter_opt",
            direction="minimize",
            sampler=sampler,
    )
    study_en.optimize(objective_en, n_trials=30)
    print("Best Score:", study_en.best_value)
    print("Best trial:", study_en.best_trial.params)
    en_scores.append(study_en.best_value)
    
    model = ElasticNet(**study_en.best_params, random_state=42)
    model.fit(train[cols], np.log1p(y))
    print('{} model training is completed'.format(i))
    ens.append(model)


hbs = []
hb_scores = []
for i, (train, cols, y)in enumerate(zip(df_trains_scaled, modeling_col_lst, targets)):
    def objective_HB(trial):
        param = {
            'epsilon':trial.suggest_float("epsilon",1.0,20.0),
            'alpha':trial.suggest_float("alpha",1e-4,20),
            'fit_intercept':trial.suggest_categorical('fit_intercept', [True, False]),
            'tol':trial.suggest_float("tol",1e-6,1e-2),
            'max_iter':1000
        }
        X_hb = train[cols]
        y_hb = np.log1p(y)

        model = HuberRegressor(**param, random_state=42)
        loo = LeaveOneOut()
        scores = cross_val_score(model, X_hb, y_hb, cv=loo, scoring='neg_mean_squared_error')
        scores = np.sqrt(-scores)
        print(f'CV scores for {i}: {scores}')
        print('Mean score : ', np.mean(scores))
        rmsle_val = np.mean(scores)

        return rmsle_val
    
    sampler = TPESampler(seed=42)
    study_hb = optuna.create_study(
            study_name="hb_parameter_opt",
            direction="minimize",
            sampler=sampler,
    )
    study_hb.optimize(objective_HB, n_trials=30)
    print("Best Score:", study_hb.best_value)
    print("Best trial:", study_hb.best_trial.params)
    hb_scores.append(study_hb.best_value)
    
    model = ElasticNet(**study_en.best_params, random_state=42)
    model.fit(train[cols], np.log1p(y))
    print('{} model training is completed'.format(i))
    hbs.append(model)


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from ngboost import NGBRegressor
from ngboost.learners import default_tree_learner
from ngboost.distns import Normal
from ngboost.scores import MLE, CRPS

ngbs = []
ngb_scores= []
for i, (train, num_f) in enumerate(zip(df_trains, num_features_lst)):
    def objective_NGB(trial):
        param = {
            "random_state":42,
#             'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.5),
            "n_estimators":trial.suggest_int("n_estimators", 100, 1000),
            "col_sample":trial.suggest_float("col_sample", 0.2, 1.0),
            'natural_gradient':trial.suggest_categorical("natural_gradient", [True,False]),
            'verbose_eval':trial.suggest_int("verbose_eval", 10, 100)
        }
        X = train[num_f]
        y = np.log1p(train['y'])

        model = NGBRegressor(Base=default_tree_learner, Dist=Normal,Score=MLE, **param)
        cv = KFold(5, shuffle=True, random_state=0)
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error', error_score='raise')
        scores = np.sqrt(-scores)
        print(f'CV scores for {i}: {scores}')
        print('Mean score : ', np.mean(scores))
        rmsle_val = np.mean(scores)

        return rmsle_val
    
    sampler = TPESampler(seed=42)
    study_ngb = optuna.create_study(
            study_name="ngb_parameter_opt",
            direction="minimize",
            sampler=sampler,
            )
    study_ngb.optimize(objective_NGB, n_trials=10)
    print("Best Score:", study_ngb.best_value)
    print("Best trial:", study_ngb.best_trial.params)
    ngb_scores.append(study_ngb.best_value)
    
    model = NGBRegressor(Base=default_tree_learner, Dist=Normal,Score=MLE,**study_ngb.best_params)
    model.fit(train[num_f], np.log1p(train['y']))
    print('{}th model training is completed'.format(i+1))
    ngbs.append(model)
 ens 
param = {
            'alpha':trial.suggest_float("alpha",1e-4,20),
            'fit_intercept':trial.suggest_categorical('fit_intercept', [True, False]),
            'normalize':trial.suggest_categorical('normalize', [True, False]),
            'tol':trial.suggest_float("tol",1e-6,1e-2),
            'selection':trial.suggest_categorical('selection', ['cyclic','random']),
            'l1_ratio':trial.suggest_float("l1_ratio",1e-6,1.0)
        }
 ridge       
param = {
          "random_state":42,
            'alpha':trial.suggest_float("alpha",0.1,10),
            'fit_intercept':trial.suggest_categorical('fit_intercept', [True, False]),
            'normalize':trial.suggest_categorical('normalize', [True, False]),
        }     
        
        
br
def objective_BR(trial):
        param = {
            'n_iter':trial.suggest_int("n_iter",10,1000),
            'alpha_2':trial.suggest_float("alpha_2",1e-6, 20),
            'lambda_2' :trial.suggest_float('lambda_2', 1e-6, 20),
            'fit_intercept':trial.suggest_categorical('fit_intercept', [True, False]),
            'normalize':trial.suggest_categorical('normalize', [True, False]),
            'tol':trial.suggest_float("tol",1e-6,1e-2)
        }
        






바꿀꺼

''' Cyclic Transformation 적용 '''
def cyclic_transformation(df, cols):
    for col in cols:
        step = col[:2]
        df[col] = pd.to_datetime(df[col])
        df[step+'_'+'month'] = df[col].dt.month
        df[step+'_'+'day'] = df[col].dt.day
        df[step+'_'+'hour'] = df[col].dt.hour
        df[step+'_'+'minute'] = df[col].dt.minute
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
        
two_trend_para = ['06_epd_para4','20_epd_para4','04_hv_para45','04_hv_para47','04_hv_para56','06_power_para57']

df_train['06_epd_para4_test'] = df_train['06_epd_para4'].apply(lambda x: 1 if x > 50 else 0)
df_train['20_epd_para4_test'] = df_train['20_epd_para4'].apply(lambda x: 1 if x < 900 else 0)
df_train['04_hv_para45_test'] = df_train['04_hv_para45'].apply(lambda x: 1 if x < 150 else 0)
df_train['04_hv_para47_test'] = df_train['04_hv_para47'].apply(lambda x: 1 if x < 100 else 0)
df_train['04_hv_para56_test'] = df_train['04_hv_para56'].apply(lambda x: 1 if x < 0.15 else 0)
df_train['06_power_para57_test'] = df_train['06_power_para57'].apply(lambda x: 1 if x > 2300 else 0)
df_train['06_power_para76_test'] = df_train['06_power_para76'].apply(lambda x: 1 if x > 1600 else 0)

df_predict['06_epd_para4_test'] = df_predict['06_epd_para4'].apply(lambda x: 1 if x > 50 else 0)
df_predict['20_epd_para4_test'] = df_predict['20_epd_para4'].apply(lambda x: 1 if x < 900 else 0)
df_predict['04_hv_para45_test'] = df_predict['04_hv_para45'].apply(lambda x: 1 if x < 150 else 0)
df_predict['04_hv_para47_test'] = df_predict['04_hv_para47'].apply(lambda x: 1 if x < 100 else 0)
df_predict['04_hv_para56_test'] = df_predict['04_hv_para56'].apply(lambda x: 1 if x < 0.15 else 0)
df_predict['06_power_para57_test'] = df_predict['06_power_para57'].apply(lambda x: 1 if x > 2300 else 0)
df_predict['06_power_para76_test'] = df_predict['06_power_para76'].apply(lambda x: 1 if x > 1600 else 0)

''' 5000 이상은 1, 아래는 0으로인코딩 '''
time_5000 = ['time para16','time_para42','time_para43','time_para44','time_para62','time_para75','time_para77','time_para89']
for col in time_5000:
    col_ = df_train.filter(regex=col+'$').columns.tolist()
    for column in col_:
        df_train[column+'_test'] = df_train[column].apply(lambda x: 1 if x>5000 else 0)
        df_predict[column+'_test'] = df_predict[column].apply(lambda x: 1 if x>5000 else 0)
''' 125 이상은 0, 아래는 1로 인코딩 '''
time_125 = ['12_time_para5','13_time_para5','17_time_para5','18_time_para5']
for col in time_125:
    df_train[col+'_test'] = df_train[col].apply(lambda x: 1 if x <125 else 0)
    df_predict[col+'_test'] = df_predict[col].apply(lambda x: 1 if x <125 else 0)
''' 3500 이상은 1 아래는 0 '''
for col in df_train.filter(regex='time_para67$').columns.tolist():
    df_train[col+'_test'] = df_train[col].apply(lambda x: 1 if x >= 3500 else 0)
    df_predict[col+'_test'] = df_predict[col].apply(lambda x: 1 if x >= 3500 else 0)
    
''' 2.6 이상 1, 아래 0 '''
'04_tmp_para31'
df_train['04_tmp_para31_test'] = df_train['04_tmp_para31'].apply(lambda x: 1 if x >=2.6 else 0)
df_predict['04_tmp_para31_test'] = df_predict['04_tmp_para31'].apply(lambda x: 1 if x >=2.6 else 0)
''' 2.8 이상 1, 아래 0 '''
'06_tmp_para31'
df_train['06_tmp_para31_test'] = df_train['06_tmp_para31'].apply(lambda x: 1 if x >=2.8 else 0)
df_predict['06_tmp_para31_test'] = df_predict['06_tmp_para31'].apply(lambda x: 1 if x >=2.8 else 0)
''' 4.0 이상 1, 아래 0 '''
tmp_4 = ['12_tmp_para31','13_tmp_para31','17_tmp_para31','18_tmp_para31','20_tmp_para31']
for col in tmp_4:
    df_train[col+'_test'] = df_train[col].apply(lambda x: 1 if x>=4.0 else 0)
    df_predict[col+'_test'] = df_predict[col].apply(lambda x: 1 if x>=4.0 else 0)
    
df_final = df_train.copy()
df_predict_final = df_predict.copy()

module_unique = df_final['module_name'].unique()
df_trains = [df_final[df_final['module_name']==eq] for eq in module_unique]
num_features_lst = []
df_predicts = [df_predict_final[df_predict_final['module_name']==eq] for eq in module_unique]
''' 중복되는 열 제거하기. '''
for i, (trains,predicts) in enumerate(zip(df_trains,df_predicts)):
    drop_col = []
    for para in for_col_filter:
        col = trains.filter(regex='^'+para).columns.tolist()
        duplicate_deleted_df = trains[col].T.drop_duplicates(subset=trains[col].T.columns, keep='first').T
        if len(trains[col].columns.difference(duplicate_deleted_df.columns))==0:  # 다른게 없으면 무시,
            continue
        else:
            drop_col.extend(trains[col].columns.difference(duplicate_deleted_df.columns).tolist())
    
    # 새로 생성한 TEST COLUMNS 전처리
#     test_col = trains.filter(regex='test$').columns.tolist()
#     duplicate_deleted_test_df = trains[test_col].T.drop_duplicates(subset=trains[test_col].T.columns, keep='first').T
#     if len(trains[test_col].columns.difference(duplicate_deleted_test_df.columns))!=0:
#         drop_col.extend(trains[test_col].columns.difference(duplicate_deleted_test_df.columns).tolist())
    
    trains.drop(drop_col,axis=1,inplace=True)
    predicts.drop(drop_col, axis=1, inplace=True)
    
#     ''' 1부터도 log transformation시 skewness와 kurtois가 많이 줄어듦. 효과적일 것이라 변환진행 ex) '20_time_para42' '''
#     skew1_df = ((trains.skew()>=2.5)|(trains.skew()<=-2.5)).reset_index().iloc[1:].reset_index(drop=True)
#     skew1_df.columns=['param','boolean']
#     high_skew1_col = skew1_df.loc[skew1_df['boolean'],:]['param'].unique().tolist()
#     df = trains[high_skew1_col]
#     ''' 04_fr_para28 , 06_fr_para28 음수를 가진 col인데, skew는 높지만 시각화시 괜찮아서 제외. '''
    
#     minus_col = df[(np.log1p(df).isnull())|(np.log1p(df)==float('-inf'))].dropna(axis=1,how='any').columns.tolist()
#     high_skew1_col = [x for x in high_skew1_col if x not in minus_col]
#     print(len(high_skew1_col))

#     trains[high_skew1_col] = np.log1p(trains[high_skew1_col])
#     predicts[high_skew1_col] = np.log1p(predicts[high_skew1_col])
    
#     assert predicts[high_skew1_col].isnull().sum().sum() == 0
#     assert (predicts[high_skew1_col]==float('-inf')).sum().sum() == 0
#     assert (predicts[high_skew1_col]==float('inf')).sum().sum() == 0
    
    var0_cols = trains.loc[:,trains.nunique()==1].columns.tolist()
    print(f'module{i}의 drop할 columns : {var0_cols}')
    trains.drop(var0_cols, axis=1, inplace=True)
    predicts.drop(var0_cols, axis=1, inplace=True)
    
    num_features = list(trains.columns[trains.dtypes==float])
    num_features.remove('y')
    scaler = StandardScaler()
    trains.loc[:, num_features] = scaler.fit_transform(trains[num_features])
    predicts.loc[:, num_features] = scaler.transform(predicts[num_features])
    
    df_trains[i] = trains
    df_predicts[i] = predicts
    
    ''' Cyclic Transformation 된 time만 사용. gen+float f들 '''
    num_features = list(trains.columns[trains.dtypes==float])
#     test_cols = trains.filter(regex='test$').columns.tolist()
    num_features.remove('y')
#     num_features_lst.append(num_features+test_cols)
    num_features_lst.append(num_features)
    
    
