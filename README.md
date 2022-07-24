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


        
