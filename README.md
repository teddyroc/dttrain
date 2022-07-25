# 챔버별 선형관계인 gen col 찾아내기 시각화
fig = plt.figure(figsize = (15, 25))
plt.style.use('dark_background')
sns.lmplot(data = df_train, x = 'gen_tmdiff_1820', y = 'y', col = 'module_name', height = 2.5, aspect = 1,
          col_wrap = 4, sharey = False, sharex = False, line_kws={'color':'red'})
plt.xticks(color='white')
plt.yticks(color='white')
plt.show();

# 추가할 para : 추가해야 될 module_name
diff_tot = [4,5,6,7,8,10,12,14,15,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,37,40,45,46]
diff_0406 = [3,4,7,8,10,11,12,13,14,15,17,19,20,21,22,24,28,31,32,33,34,35,37,44]
diff_0612 = [4,5,6,7,8,9,10,12,13,14,15,21,23,24,25,26,30,32,33,34,35,37,40,45]
diff_1317 = [2,3,5,6,7,12,18,20,25,27,29,31,32,33,37,39]
diff_1820 = [0,1,2,5,9,10,12,14,17,18,20,21,23,24,25,32,33,35,36,38,42,43,45]

# 챔버별 추가하기 함수.
''' steps는 string으로 준다. '''
def gen_tmdiff(df, steps=None):
    if steps == None:
        df['gen_tmdiff'] = (df['20_end_time'] - df['04_end_time']).dt.total_seconds()
        return df
    else:
        pre_step = steps[:2]
        post_step = steps[2:]
        df[f'gen_tmdiff_{steps}'] = (df[f'{post_step}_end_time'] - df[f'{pre_step}_end_time']).dt.total_seconds()
        return df

만약 for문 돌때.
for i, train in enumerate(df_trains):
    if i in diff_tot:
        train = gen_tmdiff(train)
        df_trains[i] = train
    if i in diff_0406:
        train = gen_tmdiff(train, '0406')
        df_trains[i] = train
    if i in diff_0612:
        train = gen_tmdiff(train, '0612')
        df_trains[i] = train
    if i in diff_1317:
        train = gen_tmdiff(train, '1317')
        df_trains[i] = train
    if i in diff_1820:
        train = gen_tmdiff(train, '1820')
        df_trains[i] = train
        
        
# EFEM 파라는 all 삭제
# gas는 우선 np.round(1)까지 해서 삭제되는 애들 시각화해보고 괜찮으면 삭제.
np.round(df_train.filter(regex='gas'), 1).loc[:,np.round(df_train.filter(regex='gas'), 1).nunique()==1]
plt.figure(figsize=(15,8))
sns.lineplot(data=df_train, x='04_end_time',y='04_gas_para74')

# 이상치 치환 1,99프로 할지 생각해보기 / 95프로가 괜찮긴했음.

# np.round 처리
- 이걸 진짜 round해서 할건지. 아니면 round해서 무의미한 para만 제거하고 끝낼건지.
RPM_COLS = df_prep_train.filter(regex='efem_para2').columns.tolist()
DIFFPRESS_COLS = df_prep_train.filter(regex='efem_para78$').columns.tolist()
POWER_COLS = df_prep_train.filter(regex='power').columns.tolist()
POSITION_COLS = df_prep_train.filter(regex='position').columns.tolist()
EPD_COLS = df_prep_train.filter(regex='epd').columns.tolist()
ESC_COLS = df_prep_train.filter(regex='esc_para')[df_prep_train.filter(regex='esc_para')>=1].dropna(how='any',axis=1).columns.tolist()  # esc_para94 제외.
ESC94_COLS = df_prep_train.filter(regex='esc_para94$').columns.tolist()
FR_COLS = df_prep_train.filter(regex='fr_para28$').columns.tolist() + df_prep_train.filter(regex='fr_para69$').columns.tolist()
FR0_COLS = df_prep_train.filter(regex='fr_para35$').columns.tolist() + df_prep_train.filter(regex='fr_para61$').columns.tolist()
GAS_COLS = df_prep_train.filter(regex='gas').columns.tolist()
HE_COLS = df_prep_train.filter(regex='he').columns.tolist()
HV_COLS = df_prep_train.filter(regex='hv').columns.tolist()
PRESS_COLS = df_prep_train.filter(regex='pressure').columns.tolist()
TEMP_COLS = df_prep_train.filter(regex='temp').columns.tolist()
TIME_COLS = df_prep_train.filter(regex='time').loc[:,df_prep_train.filter(regex='time').dtypes==float].columns.tolist()    # END_TIME 제외
TMP_COLS = df_prep_train.filter(regex='tmp').columns.tolist()

df_prep_train.loc[:, RPM_COLS] = np.round(df_prep_train[RPM_COLS],0)
df_prep_train.loc[:, DIFFPRESS_COLS] = np.round(df_prep_train[DIFFPRESS_COLS],1)
df_prep_train.loc[:, POWER_COLS] = np.round(df_prep_train[POWER_COLS],0)
df_prep_train.loc[:, POSITION_COLS] = np.round(df_prep_train[POSITION_COLS],1)
df_prep_train.loc[:, EPD_COLS] = np.round(df_prep_train[EPD_COLS],1)
df_prep_train.loc[:, ESC_COLS] = np.round(df_prep_train[ESC_COLS],0)
df_prep_train.loc[:, ESC94_COLS] = np.round(df_prep_train[ESC94_COLS],2)
df_prep_train.loc[:, FR_COLS] = np.round(df_prep_train[FR_COLS],0)
df_prep_train.loc[:, FR0_COLS] = np.round(df_prep_train[FR0_COLS],1)
df_prep_train.loc[:, GAS_COLS] = np.round(df_prep_train[GAS_COLS],2)
df_prep_train.loc[:, HE_COLS] = np.round(df_prep_train[HE_COLS],2)
df_prep_train.loc[:, HV_COLS] = np.round(df_prep_train[HV_COLS],1)
df_prep_train.loc[:, HE_COLS] = np.round(df_prep_train[HE_COLS],2)
df_prep_train.loc[:, PRESS_COLS] = np.round(df_prep_train[PRESS_COLS],0)
df_prep_train.loc[:, TEMP_COLS] = np.round(df_prep_train[TEMP_COLS],1)
df_prep_train.loc[:, TIME_COLS] = np.round(df_prep_train[TIME_COLS],1)
df_prep_train.loc[:, TMP_COLS] = np.round(df_prep_train[TMP_COLS],1)


        
# 0725 시작
챔버별 전처리 확인시 TIME이 STEP별로 동일한 COLUMN이 있었음. 이게 gen_tmdiff 랑 시간 비교해보고 만약 그 차이만큼 차이나는거면, 공분산성을 줄이기 위해 어느 하나를 제거하자.
만약 다르면 전처리를 step을 합쳐서 하는 수밖에.
다를시)
for_col_filter = []
for para in sensors_nm:
    for_col_filter.append(para.split('_')[0])
for_col_filter = list(set(for_col_filter))
for_col_filter

같을시)
기존 for_col_filter 그대로.

챔버 인코딩, 후 단일 컬럼 미리 제거.
# 우선 단일 columns 제거.
unique_col = df_train.loc[:,df_train.nunique()==1].columns.tolist()
df_train.drop(unique_col, axis=1, inplace=True)
df_predict.drop(unique_col, axis=1, inplace=True)

EFEM PARA 삭제
efem_cols = df_train.filter(regex='efem').columns.tolist()
df_train.drop(efem_cols, axis=1, inplace=True)
df_predict.drop(efem_cols, axis=1, inplace=True)

TIME PARA 전처리
df = df_train.filter(regex='time').drop(df_train.filter(regex='end_time$').columns.tolist(), axis=1)
time_para = []
for col in df.columns:
    time_para.append(col[3:])
time_para = sorted(list(set(time_para)))
time_para.remove('time_para5')    # 해석할 수 없는 그래프, 나중에 처리.

TIME PARA 시각화
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
        
재세정한 이력 有
''' 재세정 챔버 검증겸 시각화 '''
sns.lineplot(data = df_train[df_train['module_name']==10], x= '04_end_time', y = '04_time_para37', linewidth = 2, legend=False, marker='o')

# 우선 RF TIME이 쭉 떨어진 애들만 COL만들기
for para in time_para:
    col = f'04_{para}'
    tmp_train = df_train.groupby('module_name')[col].shift(1).fillna(0)
    tmp_predict = df_predict.groupby('module_name')[col].shift(1).fillna(0)
    df_train[f'tmp_{para}'] = df_train[col]-tmp_train
    df_predict[f'tmp_{para}'] = df_predict[col]-tmp_predict
    df_train.loc[:, f'CLN_DAY_{para}'] = df_train[f'tmp_{para}'].apply(lambda x: 1 if x<=0 else 0)   # 전체 dataset에 대해서 처리되있는 df,
    df_predict.loc[:, f'CLN_DAY_{para}'] = df_predict[f'tmp_{para}'].apply(lambda x: 1 if x<=0 else 0)
    
df_train.drop(df_train.filter(regex='^tmp').columns.tolist(), axis=1, inplace=True)
df_predict.drop(df_predict.filter(regex='^tmp').columns.tolist(), axis=1, inplace=True)

print('CLN columns added to TRAIN DATASET : {}'.format(len(df_train.filter(regex='^CLN').columns)))
print('CLN columns added to PREDICT DATASET: {}'.format(len(df_predict.filter(regex='^CLN').columns)))
df_train.filter(regex='^CLN').head(3)

# TIME PARA5 전처리
''' 125 이상은 0, 아래는 1로 인코딩(데이터수가 많은 걸 BASE(0)으로.) '''
time_125 = ['12_time_para5','13_time_para5','17_time_para5','18_time_para5']
for col in time_125:
    df_train[col+'_test'] = df_train[col].apply(lambda x: 1 if x <125 else 0)
    df_predict[col+'_test'] = df_predict[col].apply(lambda x: 1 if x <125 else 0)
    
# np.round(1) 적용
num_features = df_train.columns[df_train.dtypes==float].tolist()
num_features.remove('y')
df_train.loc[:,num_features] = np.round(df_train[num_features],1)

두줄 트렌드 TEST COL 추가
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
''' 125 이상은 0, 아래는 1로 인코딩 '''
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
    
Cyclic Transformation 적용 후 모델별

module_unique = df_final['module_name'].unique()
df_trains = [df_final[df_final['module_name']==eq] for eq in module_unique]
num_features_lst = []
df_predicts = [df_predict_final[df_predict_final['module_name']==eq] for eq in module_unique]
''' 중복되는 열 제거하기. '''
for i, (trains, predicts) in enumerate(zip(df_trains,df_predicts)):
    drop_col = []
    for para in for_col_filter+['CLN']:
        col = trains.filter(regex='^'+para).columns.tolist()
        if col:
            duplicate_deleted_df = trains[col].T.drop_duplicates(subset=trains[col].T.columns, keep='first').T
            if len(trains[col].columns.difference(duplicate_deleted_df.columns))==0:  # 다른게 없으면 무시,
                continue
            else:
                drop_col.extend(trains[col].columns.difference(duplicate_deleted_df.columns).tolist())
        else:
            continue
            
    # 새로 생성한 TEST COLUMNS 전처리
    test_col = trains.filter(regex='test$').columns.tolist()
    duplicate_deleted_test_df = trains[test_col].T.drop_duplicates(subset=trains[test_col].T.columns, keep='first').T
    if len(trains[test_col].columns.difference(duplicate_deleted_test_df.columns))!=0:
        drop_col.extend(trains[test_col].columns.difference(duplicate_deleted_test_df.columns).tolist())

    trains.drop(drop_col,axis=1,inplace=True)
    predicts.drop(drop_col, axis=1, inplace=True)
    print(f'module{i}의 drop된 중복된 columns : {drop_col} \n')
    
    ''' CLN_DAY COLUMN들 마지막 1을 기준으로 0,1 인코딩 '''
    for col in trains.filter(regex='^CLN').columns.tolist():
        if col:
            if trains[trains[col]==1].index.tolist():
                if predicts[predicts[col]==1].index.tolist():
                    CLN_INDEX_TRAIN = trains[trains[col]==1].index[-1]
                    CLN_INDEX_PREDICT = predicts[predicts[col]==1].index[-1]
                    trains.loc[:CLN_INDEX_TRAIN-1, col] = 0
                    trains.loc[CLN_INDEX_TRAIN:, col] = 1
                    predicts.loc[:CLN_INDEX_PREDICT-1, col] = 0
                    predicts.loc[CLN_INDEX_PREDICT:, col] = 1
                else:
                    CLN_INDEX_TRAIN = trains[trains[col]==1].index[-1]
                    trains.loc[:CLN_INDEX_TRAIN-1, col] = 0
                    trains.loc[CLN_INDEX_TRAIN:, col] = 1
            else:
                continue
        else:
            continue
    
    # gen_tmdiff col 챔버별 추가.
    if i in diff_tot:
        trains = gen_tmdiff(trains)
        predicts = gen_tmdiff(predicts)
    if i in diff_0406:
        trains = gen_tmdiff(trains, '0406')
        predicts = gen_tmdiff(predicts, '0406')
    if i in diff_0612:
        trains = gen_tmdiff(trains, '0612')
        predicts = gen_tmdiff(predicts, '0612')
    if i in diff_1317:
        trains = gen_tmdiff(trains, '1317')
        predicts = gen_tmdiff(predicts, '1317')
    if i in diff_1820:
        trains = gen_tmdiff(trains, '1820')
        predicts = gen_tmdiff(predicts, '1820')
    
    var0_cols = trains.loc[:,trains.nunique()==1].columns.tolist()
    print(f'module{i}의 drop할 columns : {var0_cols}')
    trains.drop(var0_cols, axis=1, inplace=True)
    predicts.drop(var0_cols, axis=1, inplace=True)
    
    df_trains[i] = trains
    df_predicts[i] = predicts
    
    ''' Cyclic Transformation 된 time만 사용. gen+float f들 '''
    num_features = list(trains.columns[trains.dtypes==float])
    CLN_COLS = trains.filter(regex='^CLN').columns.tolist()
    TEST_COLS = trains.filter(regex='test$').columns.tolist()
    num_features.remove('y')
    num_features_lst.append(num_features + CLN_COLS + TEST_COLS)
    
    



