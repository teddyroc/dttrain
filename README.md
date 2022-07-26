0726 통데이터로 돌리기
# 전체 및 개별 공정 소요시간 변수를 생성하는 함수입니다.
def gen_tmdiff(df, lst_stepsgap):
    df['gen_tmdiff'] = (df['20_end_time'] - df['04_end_time']).dt.total_seconds()
    for stepgap in lst_stepsgap:
        df[f'gen_tmdiff_{stepgap}'] = (df[f'{stepgap[2:]}_end_time'] - df[f'{stepgap[:2]}_end_time']).dt.total_seconds()
    return df
    
lst_steps 까지 처리후
df_train = gen_tmdiff(df_train, lst_stepsgap)
df_predict = gen_tmdiff(df_predict, lst_stepsgap)
df_train.filter(regex='tmdiff').head(2)

df_train.drop(['gen_tmdiff_1213','gen_tmdiff_1718'], axis=1, inplace=True)
df_predict.drop(['gen_tmdiff_1213','gen_tmdiff_1718'], axis=1, inplace=True)

for_col_filter = []
for para in sensors_nm:
    for_col_filter.append(para.split('_')[0])
for_col_filter = list(set(for_col_filter))
for_col_filter

# 챔버인코딩 후

# 우선 단일 columns 제거.
unique_col = df_train.loc[:,df_train.nunique()==1].columns.tolist()
df_train.drop(unique_col, axis=1, inplace=True)
df_predict.drop(unique_col, axis=1, inplace=True)

EFEM 삭제
efem_cols = df_train.filter(regex='efem').columns.tolist()
df_train.drop(efem_cols, axis=1, inplace=True)
df_predict.drop(efem_cols, axis=1, inplace=True)


GAS 전처리
gas_ = []
for para in df_train.filter(regex='gas').columns.tolist():
    gas_.append(para.split('_')[1]+'_'+para.split('_')[2])
gas_ = sorted(list(set(gas_)))

시각화
for filterp in gas_:
    paras = df_train.filter(regex=filterp+'$').columns.tolist()
    n_cols = len(paras)
    fig = plt.figure(figsize=(20,4*n_cols))
    for i,para in enumerate(paras):
        fig = plt.figure(figsize=(20,3*n_cols))
        plt.subplot(n_cols, 1, i+1)
        sns.lineplot(data = df_train, x= para[:2]+'_end_time', y = para, hue='module_name', linewidth = 2, legend=False)
        plt.title(f'{para.upper()}')
#         plt.subplots_adjust(hspace = 1.0)
        plt.ylabel('')
        plt.xlabel('')
        plt.show()
#         print(f'{para.upper()}의 세정 전 RF TIME : {df_train[para].max()}')

# all step 사용하는 gas
PARA21_COL = df_train.filter(regex='gas_para21$').columns.tolist()
PARA36_COL = df_train.filter(regex='gas_para36$').columns.tolist()
PARA6_COL = df_train.filter(regex='gas_para6$').columns.tolist()
ROUND2_GAS_COL = PARA21_COL+PARA36_COL+PARA6_COL+['20_gas_para39','13_gas_para46','18_gas_para46','04_gas_para51']

GAS_COL = df_train.filter(regex='gas').columns.tolist()
GAS_COL = [col for col in GAS_COL if col not in ROUND2_GAS_COL]
df_train.drop(GAS_COL, axis=1, inplace=True)

# TEMP 전처리
temp_ = []
for para in df_train.filter(regex='temp').columns.tolist():
    temp_.append(para.split('_')[1]+'_'+para.split('_')[2])
temp_ = sorted(list(set(temp_)))

'''시각화'''
for filterp in temp_:
    paras = df_train.filter(regex=filterp+'$').columns.tolist()
    n_cols = len(paras)
    fig = plt.figure(figsize=(20,4*n_cols))
    for i,para in enumerate(paras):
        fig = plt.figure(figsize=(20,3*n_cols))
        plt.subplot(n_cols, 1, i+1)
        sns.lineplot(data = df_train, x= para[:2]+'_end_time', y = para, hue='module_name', linewidth = 2, legend=False, marker='o')
        plt.title(f'{para.upper()}')
#         plt.subplots_adjust(hspace = 1.0)
        plt.ylabel('')
        plt.xlabel('')
        plt.show()
#         print(f'{para.upper()}의 세정 전 RF TIME : {df_train[para].max()}')

ALL_DROP = ['temp_para11', 'temp_para23', 'temp_para32', 'temp_para53', 'temp_para55', 'temp_para79', 'temp_para87', 'temp_para92', 'temp_para93','04_temp_para17', '06_temp_para17']
for col in ALL_DROP:
    cols = df_train.filter(regex=col+'$').columns.tolist()
    df_train.drop(cols,axis=1,inplace=True)
    df_predict.drop(cols,axis=1,inplace=True)

# time_para66이 1도정도 한챔버가 튀었는데, 삭제할지 말지, 처음엔 삭제안한게 더 높았는데, 모델수정 후에는 삭제한게 더 높다.
    
TIME PARA 전처리
df = df_train.filter(regex='time').drop(df_train.filter(regex='end_time$').columns.tolist(), axis=1)
time_para = []
for col in df.columns:
    time_para.append(col[3:])
time_para = sorted(list(set(time_para)))
time_para.remove('time_para5')    # 나중에 처리.

for filterp in time_para:
    paras = df.filter(regex=filterp+'$').columns.tolist()
    n_cols = len(paras)
    fig = plt.figure(figsize=(20,4*n_cols))
    for i,para in enumerate(paras):
        fig = plt.figure(figsize=(20,3*n_cols))
        plt.subplot(n_cols, 1, i+1)
        sns.lineplot(data = df_train, x= para[:2]+'_end_time', y = para, hue='module_name', linewidth = 2, legend=False)
        plt.axhline(df_train[para].max(), color='r',linewidth=2)
        plt.title(f'{para.upper()}')
#         plt.subplots_adjust(hspace = 1.0)
        plt.ylabel('')
        plt.xlabel('')
        plt.show()
        print(f'{para.upper()}의 세정 전 RF TIME : {df_train[para].max()}')
        
# 우선 RF TIME이 쭉 떨어진 애들만 COL만들기
for para in time_para:
    col = f'04_{para}'
    tmp_train = df_train.groupby('module_name')[col].shift(1).fillna(0)
    tmp_predict = df_predict.groupby('module_name')[col].shift(1).fillna(0)
    df_train[f'tmp_{para}'] = df_train[col]-tmp_train
    df_predict[f'tmp_{para}'] = df_predict[col]-tmp_predict
    df_train.loc[:, f'CLN_DAY_{para}'] = df_train[f'tmp_{para}'].apply(lambda x: 1 if x<=0 else 0)   # 전체 dataset에 대해서 처리돼있는 df,
    df_predict.loc[:, f'CLN_DAY_{para}'] = df_predict[f'tmp_{para}'].apply(lambda x: 1 if x<=0 else 0)
df_train.drop(df_train.filter(regex='^tmp').columns.tolist(), axis=1, inplace=True)
df_predict.drop(df_predict.filter(regex='^tmp').columns.tolist(), axis=1, inplace=True)

print('CLN columns added to TRAIN DATASET : {}'.format(len(df_train.filter(regex='^CLN').columns)))
print('CLN columns added to PREDICT DATASET: {}'.format(len(df_predict.filter(regex='^CLN').columns)))
df_train.filter(regex='^CLN').head(3)

# 두줄 트렌드 TEST COL 추가
df_train['06_epd_para4_test'] = df_train['06_epd_para4'].apply(lambda x: 1 if x > 50 else 0)
df_train['20_epd_para4_test'] = df_train['20_epd_para4'].apply(lambda x: 1 if x < 900 else 0)
df_train['04_hv_para45_test'] = df_train['04_hv_para45'].apply(lambda x: 1 if x < 150 else 0)
df_train['04_hv_para47_test'] = df_train['04_hv_para47'].apply(lambda x: 1 if x < 80 else 0)
df_train['04_hv_para56_test'] = df_train['04_hv_para56'].apply(lambda x: 1 if x < 0.1 else 0)
df_train['06_power_para57_test'] = df_train['06_power_para57'].apply(lambda x: 1 if x > 2250 else 0)
df_train['06_power_para76_test'] = df_train['06_power_para76'].apply(lambda x: 1 if x > 1600 else 0)

df_predict['06_epd_para4_test'] = df_predict['06_epd_para4'].apply(lambda x: 1 if x > 50 else 0)
df_predict['20_epd_para4_test'] = df_predict['20_epd_para4'].apply(lambda x: 1 if x < 900 else 0)
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
time_125 = ['12_time_para5','13_time_para5','17_time_para5','18_time_para5']
for col in time_125:
    df_train[col+'_test'] = df_train[col].apply(lambda x: 1 if x <125 else 0)
    df_predict[col+'_test'] = df_predict[col].apply(lambda x: 1 if x <125 else 0)
    
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
    
# np.round 하기
TIME5_COL = df_train.filter(regex='time_para5$').columns.tolist()
TOT_TCOL = df_train.filter(regex='time').loc[:,df_train.filter(regex='time').dtypes==float].columns.tolist()
TOT_TCOL = [col for col in TOT_TCOL if col not in TIME5_COL]
df_train.loc[:, TOT_TCOL] = np.round(df_train[TOT_TCOL],0)    # TIME PARA5를 제외한 나머지만 round(0)
df_predict.loc[:, TOT_TCOL] = np.round(df_predict[TOT_TCOL],0)

df_train.loc[:, ROUND2_GAS_COL] = np.round(df_train[ROUND2_GAS_COL],2)
df_predict.loc[:, ROUND2_GAS_COL] = np.round(df_predict[ROUND2_GAS_COL],2)

num_features = df_train.columns[df_train.dtypes==float].tolist()
num_features.remove('y')
num_features = [col for col in num_features if col not in ROUND2_GAS_COL]
df_train.loc[:,num_features] = np.round(df_train[num_features],1)    # TIME PARA는 round(1), TEMP도 round(1)
df_predict.loc[:,num_features] = np.round(df_predict[num_features],1)

C.T 하고
df_final = df_train.copy()
df_predict_final = df_predict.copy()

''' 중복열 제거 step, para number 상관없이. '''
drop_col = []
for para in for_col_filter+['CLN']:
    col = df_final.filter(regex=para).columns.tolist()
    if col:
        duplicate_deleted_df = df_final[col].T.drop_duplicates(subset=df_final[col].T.columns, keep='first').T
        if len(df_final[col].columns.difference(duplicate_deleted_df.columns))==0:  # 다른게 없으면 무시,
            continue
        else:
            drop_col.extend(df_final[col].columns.difference(duplicate_deleted_df.columns).tolist())
    else:
        continue
            
# 새로 생성한 TEST COLUMNS 전처리
test_col = df_final.filter(regex='test$').columns.tolist()
duplicate_deleted_test_df = df_final[test_col].T.drop_duplicates(subset=df_final[test_col].T.columns, keep='first').T
if len(df_final[test_col].columns.difference(duplicate_deleted_test_df.columns))!=0:
    drop_col.extend(df_final[test_col].columns.difference(duplicate_deleted_test_df.columns).tolist())
    
drop_col = list(set(drop_col))
df_final.drop(drop_col, axis=1, inplace=True)
df_predict_final.drop(drop_col, axis=1, inplace=True)
print(f'중복돼 제거된 COLUMNS : {drop_col}')

var0_col = df_final.loc[:,df_final.nunique()==1].columns.tolist()
df_final.drop(var0_col, axis=1, inplace=True)
df_predict_final.drop(var0_col, axis=1, inplace=True)

for i in range(47):
    trains = df_final[df_final['module_name']==i]
    predicts = df_predict_final[df_predict_final['module_name']==i]
    ''' CLN_DAY COLUMN들 마지막 1을 기준으로 0,1 인코딩 '''
    for col in trains.filter(regex='^CLN').columns.tolist():
        if col:
            if trains[trains[col]==1].index.tolist():
                if predicts[predicts[col]==1].index.tolist():
                    CLN_INDEX_TRAIN = trains[trains[col]==1].index[-1]    # 마지막 index를 기준으로, 
                    CLN_INDEX_PREDICT = predicts[predicts[col]==1].index[-1]
                    trains.loc[:CLN_INDEX_TRAIN-1, col] = 0
                    trains.loc[CLN_INDEX_TRAIN:, col] = 1
                    predicts.loc[:CLN_INDEX_PREDICT-1, col] = 0
                    predicts.loc[CLN_INDEX_PREDICT:, col] = 1
                    df_final[df_final['module_name']==i] = trains
                    df_predict_final[df_predict_final['module_name']==i] = predicts
                else:
                    CLN_INDEX_TRAIN = trains[trains[col]==1].index[-1]
                    trains.loc[:CLN_INDEX_TRAIN-1, col] = 0
                    trains.loc[CLN_INDEX_TRAIN:, col] = 1
                    df_final[df_final['module_name']==i] = trains
            else:
                continue
        else:
            continue

# module name 원핫인코딩
df_final_ohe = pd.concat([df_final, pd.get_dummies(df_final['module_name'], prefix='module_name')],axis=1)
df_predict_final_ohe = pd.concat([df_predict_final, pd.get_dummies(df_predict_final['module_name'], prefix='module_name')],axis=1)
df_final_ohe.drop('module_name', axis=1, inplace=True)
df_predict_final_ohe.drop('module_name', axis=1, inplace=True)
df_final_ohe.head()

''' Cyclic Transformation 된 time만 사용. gen+float f들 '''
# CAT 용
num_features = list(df_final.columns[trains.dtypes==float])
num_features.remove('y')
CLN_COLS = df_final.filter(regex='^CLN').columns.tolist()
TEST_COLS = df_final.filter(regex='test$').columns.tolist()
CAT_FEATURES = ['module_name']

COLS = CAT_FEATURES + num_features + CLN_COLS + TEST_COLS
# XGB 용
MODULE_ohe = df_final_ohe.filter(regex='^module_name').columns.tolist()

COLS_ohe = MODULE_ohe + num_features + CLN_COLS + TEST_COLS


# LGB
def objective_LGB(trial):
    param_lgb = {
            'objective':'regression',
            'metric':'rmse',
            "random_state":42,
            'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.5, step=0.01),
#             "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 4e-5),
#             "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 9e-2),
            'feature_fraction' :trial.suggest_float('feature_fraction', 0.1, 1.0, step=0.1),
            "n_estimators":trial.suggest_int("n_estimators", 100, 2000, step=10),
            "max_depth":-1,
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "max_bin": trial.suggest_int("max_bin", 100, 500)
    }
    X = df_final[COLS]
    y = df_final['y']

    model = lgb.LGBMRegressor(**param_lgb, categorical_feature=0)
    cv = KFold(11, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    scores = np.sqrt(-scores)
    print(f'CV scores : {scores}')
    print('Mean score : ', np.mean(scores))
    rmsle_val = np.mean(scores)
     
    return rmsle_val
    
sampler = TPESampler(seed=42)
study_lgb = optuna.create_study(
                study_name="lgb_parameter_opt",
                direction="minimize",
                sampler=sampler,
            )

study_lgb.optimize(objective_LGB, n_trials=30)
print("Best Score:", study_lgb.best_value)
print("Best trial:", study_lgb.best_trial.params)
    
model = lgb.LGBMRegressor(**study_lgb.best_params, objective='regression', metric='rmse', random_state=42, categorical_feature=0)
model.fit(df_final[COLS], df_final['y'])
print('model training is completed')
>> 6.881457










