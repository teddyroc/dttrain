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

FR 전처리
PARA28_COLS = df_train.filter(regex='fr_para28$').columns.tolist()
PARA28_COLS = [col for col in PARA28_COLS if col not in ['18_fr_para28']]
PARA69_COLS = df_train.filter(regex='fr_para69$').columns.tolist()
PARA69_COLS = [col for col in PARA69_COLS if col not in ['06_fr_para69','13_fr_para69','18_fr_para69']]
df_train.drop(PARA28_COLS+PARA69_COLS+['04_fr_para35','06_fr_para35'], axis=1, inplace=True)

HV 전처리
TMP 전처리
POSITION 전처리
PRESSURE 전처리

ESC 전처리
PARA84_COLS = df_train.filter(regex='esc_para84$').columns.tolist()
PARA84_COLS = [col for col in PARA84_COLS if col not in ['04_esc_para84','13_esc_para84','18_esc_para84']]
df_train.drop(PARA84_COLS, axis=1, inplace=True)
df_predict.drop(PARA84_COLS, axis=1, inplace=True)

POWER 전처리
PWR_PARA68 = df_train.filter(regex='power_para68$').columns.tolist()
PWR_PARA57 = df_train.filter(regex='04_power_para57$').columns.tolist()
# PWR_PARA82 = df_train.filter(regex='power_para82$').columns.tolist()
# PWR_PARA49 = df_train.filter(regex='power_para49$').columns.tolist()
DROP_PWR = PWR_PARA68+PWR_PARA57

df_train.drop(DROP_PWR, axis=1, inplace=True)
df_predict.drop(DROP_PWR, axis=1, inplace=True)

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
    
GAS_PARA13 = ['12_gas_para13','17_gas_para13','20_gas_para13']
GAS_PARA15 = df_train.filter(regex='gas_para15$').columns.tolist()
GAS_PARA15 = [col for col in GAS_PARA15 if col not in ['04_gas_para15']]
GAS_PARA27 = ['04_gas_para27','06_gas_para27','20_gas_para27']
# GAS_PARA33 = df_train.filter(regex='gas_para33$').columns.tolist()   애매하다,
# GAS_PARA33 = [col for col in GAS_PARA33 if col not in ['20_gas_para33']]
GAS_PARA39 = df_train.filter(regex='gas_para39$').columns.tolist()
GAS_PARA39 = [col for col in GAS_PARA39 if col not in ['20_gas_para39']]
GAS_PARA46 = ['04_gas_para46','06_gas_para46','06_gas_para46']
GAS_PARA50 = df_train.filter(regex='gas_para50$').columns.tolist()
GAS_PARA50 = [col for col in GAS_PARA50 if col not in ['20_gas_para50']]
GAS_PARA51 = ['06_gas_para51','12_gas_para51','17_gas_para51','20_gas_para51']
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

수치 단순화
# np.round 하기
TIME5_COL = df_train.filter(regex='time_para5$').columns.tolist()
TOT_TCOL = df_train.filter(regex='time').loc[:,df_train.filter(regex='time').dtypes==float].columns.tolist()
TOT_TCOL = [col for col in TOT_TCOL if col not in TIME5_COL]
df_train.loc[:, TOT_TCOL] = np.round(df_train[TOT_TCOL],0)    # TIME PARA5를 제외한 나머지만 round(0)
df_predict.loc[:, TOT_TCOL] = np.round(df_predict[TOT_TCOL],0)

GAS_COL = df_train.filter(regex='gas').columns.tolist()
df_train.loc[:, GAS_COL] = np.round(df_train[GAS_COL],2)
df_predict.loc[:, GAS_COL] = np.round(df_predict[GAS_COL],2)

num_features = df_train.columns[df_train.dtypes==float].tolist()
num_features.remove('y')
num_features = [col for col in num_features if col not in GAS_COL]
df_train.loc[:,num_features] = np.round(df_train[num_features],1)    # TIME PARA는 round(1), TEMP도 round(1)
df_predict.loc[:,num_features] = np.round(df_predict[num_features],1)

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
        
        
LGB 돌리고서 Feature Importance 
IMP_100COLS = pd.DataFrame({'params':COLS_ohe, 'importances':model_lgb.feature_importances_}).sort_values(by='importances', ascending=False)[:200].params.tolist()






