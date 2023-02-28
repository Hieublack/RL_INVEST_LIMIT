import numpy as np
import pandas as pd
import os

from numba import njit
@njit()
def hmean(arr):
    return len(arr)/np.sum(1/arr)

@njit()
def gmean(arr):
    return np.exp(np.mean(np.log(arr)))

@njit()
def check_limit_val(profit, valuation, new_profit, limit_val_new):
    v_profit = new_profit.copy()
    for i in range(len(new_profit)-1, -1, -1):
        if valuation[i] > limit_val_new:
            v_profit[i] = profit[i]
    if gmean(v_profit) >= gmean(new_profit):
        return True
    return False
    
@njit()
def value_limit_drop(congthuc, profit, valuation):
    new_profit = np.array([profit[0]])
    limit_val = np.zeros(5)
    if profit[0] > 1.03:
        limit_val[-1] = valuation[0]
    else:
        limit_val[-1] = valuation[0]/(profit[0]**2)
    for id in range(1, len(profit)):
        #xử lí lợi nhuân
        if valuation[id] < limit_val[-1]:
            new_profit = np.append(new_profit, 1.03)
        else:
            new_profit = np.append(new_profit, profit[id])
        #xử lí ngưỡng
        if valuation[id] == 0:
            valuation[id] = gmean(valuation[: id])
        if profit[id] > 1.03 and valuation[id] <= limit_val[-1]:
            check = check_limit_val(profit, valuation, new_profit, gmean(np.append(limit_val, valuation[id]))/gmean(profit[: id])/profit[id])
            if check == True:
                limit_val = np.append(limit_val, gmean(np.append(valuation[: id+1], limit_val[0]))/gmean(profit[:id+1])/profit[id])
            else:
                limit_val = np.append(limit_val, gmean(np.append(valuation[: id+1], limit_val[0]))/gmean(profit[id:]))
        else:
            limit_val = np.append(limit_val, gmean(valuation[: id+1])/gmean(profit[: id+1]))
    # return [limit_val,gmean(new_profit[1:]), gmean(new_profit)]
    return limit_val

@njit()
def value_limit(congthuc, profit, valuation):
    profit_bank = 1.06
    new_profit = np.array([profit[0]])
    limit_val = np.full(5, np.inf)
    if profit[0] > profit_bank:
        limit_val[-1] = valuation[0]
    else:
        limit_val[-1] = valuation[0]/(profit[0]**2)
    for id in range(4, len(profit)-1):
        #xử lí lợi nhuân
        if valuation[id] < limit_val[-1]:
            new_profit = np.append(new_profit, profit_bank)
        else:
            new_profit = np.append(new_profit, profit[id])
        #xử lí ngưỡng
        if valuation[id] == 0:
            valuation[id] = gmean(valuation[: id])
        if profit[id] > profit_bank and valuation[id] <= limit_val[-1]:
            check = check_limit_val(profit, valuation, new_profit, gmean(np.append(limit_val, valuation[id]))/gmean(profit[: id])/profit[id])
            if check == True:
                limit_val = np.append(limit_val, gmean(np.append(valuation[: id+1], limit_val[0]))/gmean(profit[:id+1])/profit[id])
            else:
                limit_val = np.append(limit_val, gmean(np.append(valuation[: id+1], limit_val[0]))/gmean(profit[id:]))
        else:
            limit_val = np.append(limit_val, gmean(valuation[: id+1])/gmean(profit[: id+1]))
        # print(valuation[id], limit_val[-2:], profit[id])
    # # return [limit_val,gmean(new_profit[1:]), gmean(new_profit)]
    # print(profit)
    # print(limit_val)
    # print(valuation)
    return limit_val



from complete_method import CompleteMethod
file_path = 'Data_YearFromQuarter.csv'
data = pd.read_csv(file_path)
# data = pd.read_csv('/content/drive/MyDrive/November/POWER_METHOD/QUARTER/data_test_not_invest.csv')
pathSaveFormula = "./test"
if not os.path.exists(pathSaveFormula):
    os.mkdir(pathSaveFormula)

NUMBER_QUARTER = 62

vis = CompleteMethod(data, pathSaveFormula, NUMBER_QUARTER, "geomean")

vis.generate_formula(1, 10000, 10000)

df_fomula = vis.convert_npy_file_to_DataFrame('./test/high_profit_3_12_02_2023_20_42_00.npy')

df = df_fomula[df_fomula.invest != 'NI'].reset_index(drop= True)


df_fomula = df_fomula[2000:3000].reset_index(drop= True)

arr_congthuc = df_fomula.formula
arr_profit = np.zeros((1000, NUMBER_QUARTER))
arr_value = np.zeros((1000, NUMBER_QUARTER))
arr_limit = np.zeros((1000, NUMBER_QUARTER))
arr_rank_not_invest = np.zeros((1000, NUMBER_QUARTER))


for i in range(len(df_fomula)):
    temp = vis.explain_formula(df.formula.iloc[i], not_invest_available= True)
    arr_profit[i] = np.array(temp.Out_Profit)
    arr_value[i] = np.array(temp.Out_Value)
    arr_rank_not_invest[i] = np.array(temp.Out_Rank_NI)
    arr_limit[i] = value_limit(df_fomula.formula.iloc[i], np.array(temp.Out_Profit), np.array(temp.Out_Value))

np.save('./congthuc/congthuc.npy', arr_congthuc, allow_pickle= True)
np.save('./congthuc/all_profit.npy', arr_profit, allow_pickle= True)
np.save('./congthuc/all_value.npy', arr_value, allow_pickle= True)
np.save('./congthuc/all_limit.npy', arr_limit, allow_pickle= True)
np.save('./congthuc/arr_rank_not_invest.npy', arr_rank_not_invest, allow_pickle= True)


print('DONE')











