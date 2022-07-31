0727 최신
# 전체 및 개별 공정 소요시간 변수를 생성하는 함수입니다.
def gen_tmdiff(df, lst_stepsgap):
    df['gen_tmdiff'] = (df['20_end_time'] - df['04_end_time']).dt.total_seconds()
    for stepgap in lst_stepsgap:
        df[f'gen_tmdiff_{stepgap}'] = (df[f'{stepgap[2:]}_end_time'] - df[f'{stepgap[:2]}_end_time']).dt.total_seconds()
    return df
    
df_train = gen_tmdiff(df_train, lst_stepsgap)
df_predict = gen_tmdiff(df_predict, lst_stepsgap)
df_train.filter(regex='tmdiff').head(2)

STEP17_COLS = df_train.filter(regex='^17').columns.tolist()
STEP12_COLS = df_train.filter(regex='^12').columns.tolist()
df_train.drop(STEP17_COLS+STEP12_COLS, axis=1, inplace=True)
df_predict.drop(STEP17_COLS+STEP12_COLS, axis=1, inplace=True)
df_train.drop(['gen_tmdiff_1213','gen_tmdiff_1718'], axis=1, inplace=True)
df_predict.drop(['gen_tmdiff_1213','gen_tmdiff_1718'], axis=1, inplace=True)

unique col 제거하고.

FR 전처리
PARA28_COLS = ['04_fr_para28','13_fr_para28','20_fr_para28']
PARA35_COLS = ['06_fr_para35']
PARA69_COLS = ['04_fr_para69','20_fr_para69']
df_train.drop(PARA28_COLS+PARA69_COLS+PARA35_COLS, axis=1, inplace=True)

HV 전처리
TMP 전처리
POSITION 전처리
PRESSURE 전처리

ESC 전처리
PARA84_COLS = ['04_esc_para84','20_esc_para84']
df_train.drop(PARA84_COLS, axis=1, inplace=True)
df_predict.drop(PARA84_COLS, axis=1, inplace=True)

POWER 전처리
df_train.drop('04_power_para57', axis=1, inplace=True)
df_predict.drop('04_power_para57', axis=1, inplace=True)

HE 전처리
HE_PARA95 = df_train.filter(regex='he_para95$').columns.tolist()
df_train.drop(HE_PARA95, axis=1, inplace=True)
df_predict.drop(HE_PARA95, axis=1, inplace=True)

GAS 전처리
''' drop해야 하는 애들 '''
# all step
DROP_ALL = ['gas_para10','gas_para19','gas_para48','gas_para70']
for para in DROP_ALL:
    df_train.drop(df_train.filter(regex=para+'$').columns.tolist(),axis=1, inplace=True)
    
# GAS_PARA13 = ['12_gas_para13','17_gas_para13','20_gas_para13']
GAS_PARA13 = ['20_gas_para13']
GAS_PARA15 = df_train.filter(regex='gas_para15$').columns.tolist()
GAS_PARA15 = [col for col in GAS_PARA15 if col not in ['04_gas_para15']]
GAS_PARA27 = ['04_gas_para27','06_gas_para27','20_gas_para27']
GAS_PARA39 = df_train.filter(regex='gas_para39$').columns.tolist()
GAS_PARA39 = [col for col in GAS_PARA39 if col not in ['20_gas_para39']]
GAS_PARA46 = ['04_gas_para46','06_gas_para46','06_gas_para46']
GAS_PARA50 = df_train.filter(regex='gas_para50$').columns.tolist()
GAS_PARA50 = [col for col in GAS_PARA50 if col not in ['20_gas_para50']]
# GAS_PARA51 = ['06_gas_para51','12_gas_para51','17_gas_para51','20_gas_para51']
GAS_PARA51 = ['06_gas_para51','20_gas_para51']
GAS_PARA59 = ['04_gas_para59']
GAS_PARA71 = df_train.filter(regex='gas_para71$').columns.tolist()
GAS_PARA71 = [col for col in GAS_PARA71 if col not in ['06_gas_para71']]
GAS_PARA74 = ['06_gas_para74','13_gas_para74','18_gas_para74','20_gas_para74']
GAS_PARA85 = df_train.filter(regex='gas_para85$').columns.tolist()
GAS_PARA85 = [col for col in GAS_PARA85 if col not in ['04_gas_para85']]

GAS_DROP_COL = GAS_PARA13+GAS_PARA15+GAS_PARA27+GAS_PARA39+GAS_PARA46+GAS_PARA50+GAS_PARA51+GAS_PARA59+GAS_PARA71+GAS_PARA74+GAS_PARA85

df_train.drop(GAS_DROP_COL, axis=1, inplace=True)
df_predict.drop(GAS_DROP_COL, axis=1, inplace=True)

TEMP 전처리
DROP_TEMP = ['temp_para11', '06_temp_para53','20_temp_para53', 'temp_para79', '20_temp_para87', '06_temp_para93','20_temp_para93', '04_temp_para17', '06_temp_para17']
for col in DROP_TEMP:
    cols = df_train.filter(regex=col+'$').columns.tolist()
    df_train.drop(cols,axis=1,inplace=True)
    df_predict.drop(cols,axis=1,inplace=True)
    
TIME 전처리
CLN COL 추가

두줄 트렌드 TEST COL 추가
df_train['06_epd_para4_test'] = df_train['06_epd_para4'].apply(lambda x: 1 if x > 50 else 0)
df_train['20_epd_para4_test'] = df_train['20_epd_para4'].apply(lambda x: 1 if x < 900 else 0)
df_train['04_hv_para3_test'] = df_train['04_hv_para3'].apply(lambda x: 1 if x < 0.04 else 0)
df_train['04_hv_para45_test'] = df_train['04_hv_para45'].apply(lambda x: 1 if x < 150 else 0)
df_train['04_hv_para47_test'] = df_train['04_hv_para47'].apply(lambda x: 1 if x < 100 else 0)
df_train['04_hv_para56_test'] = df_train['04_hv_para56'].apply(lambda x: 1 if x < 0.1 else 0)
df_train['06_power_para57_test'] = df_train['06_power_para57'].apply(lambda x: 1 if x > 2250 else 0)
df_train['06_power_para76_test'] = df_train['06_power_para76'].apply(lambda x: 1 if x > 1600 else 0)

df_predict['06_epd_para4_test'] = df_predict['06_epd_para4'].apply(lambda x: 1 if x > 50 else 0)
df_predict['20_epd_para4_test'] = df_predict['20_epd_para4'].apply(lambda x: 1 if x < 900 else 0)
df_predict['04_hv_para3_test'] = df_predict['04_hv_para3'].apply(lambda x: 1 if x < 0.04 else 0)
df_predict['04_hv_para45_test'] = df_predict['04_hv_para45'].apply(lambda x: 1 if x < 150 else 0)
df_predict['04_hv_para47_test'] = df_predict['04_hv_para47'].apply(lambda x: 1 if x < 100 else 0)
df_predict['04_hv_para56_test'] = df_predict['04_hv_para56'].apply(lambda x: 1 if x < 0.1 else 0)
df_predict['06_power_para57_test'] = df_predict['06_power_para57'].apply(lambda x: 1 if x > 2300 else 0)
df_predict['06_power_para76_test'] = df_predict['06_power_para76'].apply(lambda x: 1 if x > 1600 else 0)

''' 5000 이상은 1, 아래는 0으로인코딩 '''
time_5000 = ['time_para16','time_para42','time_para43','time_para44','time_para62','time_para75','time_para77','time_para89']
CLN_COLS = df_train.filter(regex='^CLN').columns.tolist()
for col in time_5000:
    col_ = df_train.filter(regex=col+'$').columns.tolist()
    col_ = [x for x in col_ if x not in CLN_COLS]    # filter된 col 중 CLN col 제외.
    for column in col_:
        df_train[column+'_test'] = df_train[column].apply(lambda x: 1 if x>5000 else 0)
        df_predict[column+'_test'] = df_predict[column].apply(lambda x: 1 if x>5000 else 0)
        
''' TIME PARA5 처리 / 125 이상은 0, 아래는 1로 인코딩(데이터수가 많은 걸 BASE(0)으로.) '''
# time_125 = ['12_time_para5','13_time_para5','17_time_para5','18_time_para5']
time_125 = ['13_time_para5','18_time_para5']
for col in time_125:
    df_train[col+'_test'] = df_train[col].apply(lambda x: 1 if x <125 else 0)
    df_predict[col+'_test'] = df_predict[col].apply(lambda x: 1 if x <125 else 0)
    
''' 2.6 이상 1, 아래 0 tmp 전체에 대해서 한 챔버만 높게 쓰고 있다. '''
'04_tmp_para31'
df_train['04_tmp_para31_test'] = df_train['04_tmp_para31'].apply(lambda x: 1 if x >=2.6 else 0)
df_predict['04_tmp_para31_test'] = df_predict['04_tmp_para31'].apply(lambda x: 1 if x >=2.6 else 0)


수치 단순화
# np.round 하기
num_features = df_train.columns[df_train.dtypes==float].tolist()
num_features.remove('y')
df_train.loc[:,num_features] = np.round(df_train[num_features],2)
df_predict.loc[:,num_features] = np.round(df_predict[num_features],2)

CYCLIC all step end time에 적용하고



# 모델링 적용에서
''' PARA 별(position, time, efem 등) Standard Scaling / MINMAX나 ROBUST 로 바꿔서도 해보자. '''

TEST_COLS = df_final_ohe.filter(regex='test$').columns.tolist()
CLN_COLS = df_final_ohe.filter(regex='^CLN').columns.tolist()
scalers = [StandardScaler() for k in range(len(for_col_filter))]
scaled_ohe = df_final_ohe[COLS_ohe].copy(deep=True)
scaled_predict_ohe = df_predict_final_ohe[COLS_ohe].copy(deep=True)

for i, (scaler, filterp) in enumerate(zip(scalers, for_col_filter+['gen'])):
    cols = scaled_ohe.filter(regex=filterp).columns.tolist()
    cols = [col for col in cols if col not in TEST_COLS]
    if filterp == 'time':
        cols = [col for col in cols if col not in CLN_COLS]
        mean = (scaled_ohe[cols].sum().sum())/(len(cols)*len(scaled_ohe))
        std = ((scaled_ohe[cols]-mean)**2).sum().sum()/(len(cols)*len(scaled_ohe))
        scaled_ohe.loc[:, cols] = (scaled_ohe[cols]-mean)/std
        scaled_predict_ohe.loc[:, cols] = (scaled_predict_ohe[cols]-mean)/std
    else:
        mean = (scaled_ohe[cols].sum().sum())/(len(cols)*len(scaled_ohe))
        std = ((scaled_ohe[cols]-mean)**2).sum().sum()/(len(cols)*len(scaled_ohe))
        scaled_ohe.loc[:, cols] = (scaled_ohe[cols]-mean)/std
        scaled_predict_ohe.loc[:, cols] = (scaled_predict_ohe[cols]-mean)/std
        

''' Feature Selection '''
LGB 돌리고서 Feature Importance 
IMP_100COLS = pd.DataFrame({'params':COLS_ohe, 'importances':model_lgb.feature_importances_}).sort_values(by='importances', ascending=False)[:200].params.tolist()


CAT = CatBoostRegressor(learning_rate=0.16,loss_function='RMSE',eval_metric='RMSE')
cv = KFold(11, shuffle=True, random_state=42)
X = df_final_ohe[COLS_ohe]
y = df_final['y']
scores = cross_val_score(CAT, X, y, cv=cv, scoring='neg_mean_squared_error', error_score='raise')
scores = np.sqrt(-scores)
print(f'CV scores : {scores}')
print('Mean score : ', np.mean(scores))

Cat 성능 일단 확인하고나서, 
from probatus.feature_elimination import EarlyStoppingShapRFECV
cat = CatBoostRegressor()
# Run feature elimination
shap_elimination = EarlyStoppingShapRFECV(
    clf=cat, step=0.2, cv=10, scoring='neg_mean_squared_error', early_stopping_rounds=15, n_jobs=-1, eval_metric='rmse')
report = shap_elimination.fit_compute(df_final_ohe[COLS_ohe], df_final_ohe['y'], feature_perturbation="tree_path_dependent")

# Make plots
performance_plot = shap_elimination.plot()

report.loc[:, report.filter(regex='mean').columns.tolist()] = np.sqrt(-report.filter(regex='mean'))
SELECTED_COLS = report.loc[report['val_metric_mean'] == report['val_metric_mean'].min(),:]['features_set'].values.tolist()[0]
SELECTED_COLS
그리고 이 col로 다시 


lgb para
param_lgb = {
            'objective':'regression',
            'metric':'rmse',
            "random_state":42,
            'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.2, step=0.01),
#             "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 4e-5),
#             "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 9e-2),
            'feature_fraction' :trial.suggest_float('feature_fraction', 0.1, 1.0, step=0.1),
            "n_estimators":trial.suggest_int("n_estimators", 100, 2000, step=10),
            "max_depth":-1,
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "max_bin": trial.suggest_int("max_bin", 100, 500)
    }


''' 같은 PARA 별 Standard Scaling '''
''' PARA별 Standard Scaling '''
scaled_ohe = df_final_ohe[COLS_ohe].copy(deep=True)
scaled_predict_ohe = df_predict_final_ohe[COLS_ohe].copy(deep=True)

''' 같은 PARA 별만 Standard Scaling '''
for filterp in sensors_nm:
    cols = scaled_ohe.filter(regex=filterp+'$').columns.tolist()
    cols = [col for col in cols if col not in CLN_COLS]
    n_cols = len(cols)
    mean = (scaled_ohe[cols].sum().sum())/(n_cols*len(scaled_ohe))
    std = ((scaled_ohe[cols]-mean)**2).sum().sum()/(n_cols*len(scaled_ohe))
    scaled_ohe.loc[:, cols] = (scaled_ohe[cols]-mean)/std
    scaled_predict_ohe.loc[:, cols] = (scaled_predict_ohe[cols]-mean)/std
    
''' 같은 PARA 별만 MINMAX SCALING '''
''' PARA별 MinMax Scaling '''
minmax_final_ohe = df_final_ohe[COLS_ohe].copy(deep=True)
minmax_predict_ohe = df_predict_final_ohe[COLS_ohe].copy(deep=True)
                
''' 같은 PARA 별만 MINMAX Scaling '''
for filterp in sensors_nm:
    cols = minmax_final_ohe.filter(regex=filterp+'$').columns.tolist()
    cols = [col for col in cols if col not in CLN_COLS]
    min_ = minmax_final_ohe[cols].min()
    max_ = minmax_final_ohe[cols].max()
    minmax_final_ohe.loc[:,cols] = (minmax_final_ohe[cols] - min_)/(max_ - min_)
    minmax_predict_ohe.loc[:, cols] = (minmax_predict_ohe[cols] - min_)/(max_ - min_)
    
''' PCA '''
def pca_pre(datas,pred_datas, cols, n, step): 
    pca = decomposition.PCA(n_components = n) 
    pca_array = pca.fit_transform(datas[cols]) 
    pca_array_pred = pca.transform(pred_datas[cols]) 
    pca_df_train = pd.DataFrame(data = pca_array, columns = ['{0}_pca{1}'.format(step, num) for num in range(n)]) 
    pca_df_pred = pd.DataFrame(data = pca_array_pred, columns = ['{0}_pca{1}'.format(step, num) for num in range(n)]) 
    return pca_df_train, pca_df_pred
    
step_lst = ['04','06','13','18','20']
SIN_COLS = scaled_ohe.filter(regex='sin$').columns.tolist()
COS_COLS = scaled_ohe.filter(regex='cos$').columns.tolist()
TEST_COLS = scaled_ohe.filter(regex='test$').columns.tolist()

NOT_PCA_COLS = SIN_COLS+COS_COLS+TEST_COLS

''' 같은 para끼리 PCA 진행. '''
for i,col in enumerate(sensors_nm): 
    cols = scaled_ohe.filter(regex=col+'$').columns.tolist() 
    cols = list(set(cols))
    cols = [k for k in cols if k not in NOT_PCA_COLS]
    n_cols = len(cols) 
    pca = decomposition.PCA(n_components = n_cols-1) 
    pca_array = pca.fit_transform(scaled_ohe[cols])
    result = pd.DataFrame({'설명가능한 분산 비율(고윳값)':pca.explained_variance_,\
         '기여율':pca.explained_variance_ratio_},\
        index=np.array([f"pca{num+1}" for num in range(n_cols-1)]))
    result['누적기여율'] = result['기여율'].cumsum()
    if len(result.loc[result['누적기여율']>=0.9,:].index) >=1:
        try:
            n = int(result.loc[result['누적기여율']>=0.9,:].index[0][-2:])
            df, df_p = pca_pre(scaled_ohe, scaled_predict_ohe, cols, n, col)
            scaled_ohe = pd.concat([scaled_ohe, df],axis=1)
            scaled_predict_ohe = pd.concat([scaled_predict_ohe, df_p],axis=1)
            scaled_ohe.drop(cols, axis=1, inplace=True)
            scaled_predict_ohe.drop(cols, axis=1, inplace=True)
        except:
            n = int(result.loc[result['누적기여율']>=0.9,:].index[0][-1])
            df, df_p = pca_pre(scaled_ohe, scaled_predict_ohe, cols, n, col)
            scaled_ohe = pd.concat([scaled_ohe, df],axis=1)
            scaled_predict_ohe = pd.concat([scaled_predict_ohe, df_p],axis=1)
            scaled_ohe.drop(cols, axis=1, inplace=True)
            scaled_predict_ohe.drop(cols, axis=1, inplace=True)
    else:
        print(cols)


ExtraTreesRegressor + Optuna
from sklearn.ensemble import ExtraTreesRegressor

def objective_ET(trial):
    param = {
#             "device_type": trial.suggest_categorical("device_type", ['gpu']),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=10),
            'max_depth': trial.suggest_int("max_depth", 1, 12, step=1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 30),
            'max_features':trial.suggest_categorical('max_features', ['auto','sqrt','log2'])
    }
        
    X = df_final_ohe[COLS_ohe]
    y = df_final_ohe['y']

    model = ExtraTreesRegressor(**param, random_state=42)  
    cv = KFold(5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error', error_score='raise')
    scores = np.sqrt(-scores)
    print(f'CV scores : {scores}')
    print('Mean score : ', np.mean(scores))
    rmse_val = np.mean(scores)
     
    return rmse_val
    
sampler = TPESampler(seed=42)
study_et = optuna.create_study(
            study_name="et_parameter_opt",
            direction="minimize",
            sampler=sampler,
            )

study_et.optimize(objective_ET, n_trials=20)
print("Best Score:", study_et.best_value)
print("Best trial:", study_et.best_trial.params)
    
model_et = ExtraTreesRegressor(**study_et.best_params, random_state=42)
model_et.fit(df_final_ohe[COLS_ohe], df_final_ohe['y'])
print('model training is completed')


''' 챔버별 optuna '''
param_lgb = {
            'objective':'regression',
            'metric':'rmse',
            "random_state":42,
            'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.2),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 4e-5),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 9e-2),
            'bagging_fraction' :trial.suggest_loguniform('bagging_fraction', 0.01, 1.0),
            "n_estimators":trial.suggest_int("n_estimators", 100, 2000),
            "max_depth":-1,
#             "colsample_bytree":trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "max_bin": trial.suggest_int("max_bin", 100, 500)
        }
        
gbr은 이거 참고해서 돌려
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =42)
                                   
def objective_GBR(trial):
    param_gbr = {
            "loss": trial.suggest_categorical("loss", ["squared_error", "huber"]),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01,0.2),
            'max_features':trial.suggest_categorical("max_features", ["auto", "sqrt","log2"])
    }
        
    X = df_final_ohe[COLS_ohe]
    y = df_final_ohe['y']

    model = GradientBoostingRegressor(**param_gbr, random_state=0, criterion='mse')
    cv = KFold(7,shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    scores = np.sqrt(-scores)
    print(f'CV scores : {scores}')
    print('Mean score : ', np.mean(scores))
    rmse_val = np.mean(scores)
     
    return rmse_val
    
sampler = TPESampler(seed=42)
study_gbr = optuna.create_study(
            study_name="gbr_parameter_opt",
            direction="minimize",
            sampler=sampler,
            )
study_gbr.optimize(objective_GBR, n_trials=30, timeout=600)
print("Best Score:", study_gbr.best_value)
print("Best trial:", study_gbr.best_trial.params)
    
model_gbr = GradientBoostingRegressor(**study_gbr.best_params, random_state=0, criterion='mse')
model_gbr.fit(df_final_ohe[COLS_ohe], df_final_ohe['y'])
print('model training is completed')


''' Feature Selection RFECV '''
TOT_COLS = minmax_final.columns.tolist()
MODULE_COLS = minmax_final.filter(regex='^module_name').columns.tolist()
CLN_COLS = minmax_final.filter(regex='^CLN').columns.tolist()
GEN_COLS = minmax_final.filter(regex='^gen').columns.tolist()
TEST_COLS = minmax_final.filter(regex='test$').columns.tolist()
NOT_RFECV_COLS = MODULE_COLS+CLN_COLS+GEN_COLS+TEST_COLS
FOR_RFECV_COLS = [col for col in TOT_COLS if col not in NOT_RFECV_COLS]    # 내가 추가한 col들은 select 대상에서 제외.

# RFECV도 과적합될 수 있기 떄문에 미리 나눠서 fit 한다.
X_train, X_test, y_train, y_test = train_test_split(minmax_final, df_final_ohe['y'], test_size=0.2, shuffle=True, random_state=42)

model_lgb = lgb.LGBMRegressor(random_state=0)
selector_lgb = RFECV(model_lgb, step=0.2, cv=10, scoring='neg_mean_squared_error')
SELECTED_DF = pd.DataFrame(selector_lgb.fit_transform(X_train[FOR_RFECV_COLS], y_train))
X_train.reset_index(drop=True, inplace=True)
df = pd.concat([X_train[NOT_RFECV_COLS],SELECTED_DF], axis=1)
df.head()

SELECTED_DF_TEST = pd.DataFrame(selector_lgb.transform(X_test[FOR_RFECV_COLS]))
X_test.reset_index(drop=True, inplace=True)
df_test = pd.concat([X_test[NOT_RFECV_COLS],SELECTED_DF_TEST], axis=1)
df_test.head(3)

''' RFECV는 이렇게 출력이 돼서 좀 불편함. '''
for i,col in enumerate(X_train.columns):
    print('Column: %s, Selected %s, Rank: %.3f' % (col, selector_lgb.support_[i], selector_lgb.ranking_[i]))

''' Feature Selection SelectKBest '''
from sklearn.feature_selection import SelectKBest, f_regression, SelectPercentile

kb = SelectKBest(mutual_info_regression, k=300).fit(X_train[FOR_RFECV_COLS], pd.Series(y_train))
for_train_concat = pd.DataFrame(kb.transform(X_train[FOR_RFECV_COLS]))
X_train.reset_index(drop=True, inplace=True)
df_train = pd.concat([for_train_concat, X_train[NOT_RFECV_COLS]],axis=1)

X_new = pd.DataFrame(kb.transform(X_test[FOR_RFECV_COLS]))
X_test.reset_index(drop=True, inplace=True)
df_test = pd.concat([X_new, X_test[NOT_RFECV_COLS]], axis=1)
df_test.head(3)

''' optuna+xgb로 TEST set에 대해서 성능 chk '''
# train set으로만 훈련 후

def objective_XGB(trial):
    param_xgb = {
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 9e-2),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            # maximum depth of the tree, signifies complexity of the tree.
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            # minimum child weight, larger the term more conservative the tree.
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01,0.2),
            # defines how selective algorithm is.
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    }
        
    X = df_train
    y = y_train

    model = xgb.XGBRegressor(**param_xgb, booster='gbtree', random_state=0)
    cv = KFold(7, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    scores = np.sqrt(-scores)
    print(f'CV scores : {scores}')
    print('Mean score : ', np.mean(scores))
    rmse_val = np.mean(scores)
     
    return rmse_val
    
sampler = TPESampler(seed=42)
study_xgb = optuna.create_study(
            study_name="xgb_parameter_opt",
            direction="minimize",
            sampler=sampler,
    )
study_xgb.optimize(objective_XGB, n_trials=30, timeout=600)
print("Best Score:", study_xgb.best_value)
print("Best trial:", study_xgb.best_trial.params)
    
model_xgb = xgb.XGBRegressor(**study_xgb.best_params, booster='gbtree', gpu_id='0', random_state=0)
model_xgb.fit(df_train, y_train)
print('model training is completed')


xgb_pred = model_xgb.predict(df_test)
RMSE(y_test, xgb_pred)

''' 해서 괜찮으면 전체에 대해서 사용. '''
# kb는 그대로 두고.
for_concat = pd.DataFrame(kb.transform(minmax_final[FOR_RFECV_COLS]))
minmax_train = pd.concat([minmax_final[NOT_RFECV_COLS], for_concat], axis=1)
for_predict_concat = pd.DataFrame(kb.transform(minmax_predict_final[FOR_RFECV_COLS]))
minmax_predict = pd.concat([minmax_predict_final[NOT_RFECV_COLS], for_predict_concat], axis=1)
print((minmax_train.columns != minmax_predict.columns).sum())
>> 0 나오면.

def objective_XGB(trial):
    param_xgb = {
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 9e-2),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            # maximum depth of the tree, signifies complexity of the tree.
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            # minimum child weight, larger the term more conservative the tree.
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01,0.2),
            # defines how selective algorithm is.
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    }
        
    X = minmax_train
    y = df_final_ohe['y']

    model = xgb.XGBRegressor(**param_xgb, booster='gbtree', random_state=0)
    cv = KFold(7, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    scores = np.sqrt(-scores)
    print(f'CV scores : {scores}')
    print('Mean score : ', np.mean(scores))
    rmse_val = np.mean(scores)
     
    return rmse_val
    
sampler = TPESampler(seed=42)
study_xgb = optuna.create_study(
            study_name="xgb_parameter_opt",
            direction="minimize",
            sampler=sampler,
    )
study_xgb.optimize(objective_XGB, n_trials=30, timeout=600)
print("Best Score:", study_xgb.best_value)
print("Best trial:", study_xgb.best_trial.params)
    
model_xgb = xgb.XGBRegressor(**study_xgb.best_params, booster='gbtree', gpu_id='0', random_state=0)
model_xgb.fit(minmax_train, df_final_ohe['y'])
print('model training is completed')

&

xgb_pred = model_xgb.predict(minmax_predict)
xgb_pred

하면 된다.

