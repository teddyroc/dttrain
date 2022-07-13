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
