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

