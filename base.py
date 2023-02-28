import pandas as pd
import numpy as np
import os
from colorama import Fore, Style
import nopy

import warnings
warnings.filterwarnings("ignore")


class Method:
    def __init__(self, data:pd.DataFrame, path_save:str, num_training:int, profit_method:str) -> None:
        # Check các cột bắt buộc
        drop_cols = ["TIME", "PROFIT", "SYMBOL"]
        for col in drop_cols:
            if col not in data.columns:
                raise Exception(f'Thiếu cột "{col}".')

        # Check kiểu dữ liệu của cột TIME và PROFIT
        if data["TIME"].dtype != "int64":
            raise Exception(f'Kiểu dữ liệu của cột "TIME" phải là int64.')
        if data["PROFIT"].dtype != "float64":
            raise Exception(f'Kiểu dữ liệu của cột "PROFIT" phải là float64.')

        # Check cột TIME xem có tăng dần không
        if data["TIME"].diff().max() > 0:
            raise Exception(f'Dữ liệu phải được sắp xếp giảm dần theo cột "TIME".')

        # Check cột TIME xem có bị khuyết quý nào không
        min_time = np.min(data["TIME"])
        max_time = np.max(data["TIME"])
        time_unique_arr = np.unique(data["TIME"])
        for i in range(min_time, max_time):
            if i not in time_unique_arr:
                raise Exception(f'Dữ liệu bị khuyết chu kỳ "{i}".')

        # Check các cột cần được drop
        for col in data.columns:
            if col not in drop_cols and data[col].dtype == "object":
                drop_cols.append(col)

        print(Fore.YELLOW + f"Các cột không được coi là biến: {drop_cols}.", Style.RESET_ALL)

        # Kiểm tra xem path có tồn tại hay không
        if type(path_save) != str or not os.path.exists(path_save):
            raise Exception(f'Không tồn tại thư mục {path_save}/.')
        else:
            if not path_save.endswith("/") and not path_save.endswith("\\"):
                path_save += "/"

            self.path = path_save

        # Thiết lập các thuộc tính
        start_ = np.min(data["TIME"])
        last_ = start_ + num_training - 1
        self.TRAINING_DATA = data[(data["TIME"] >= start_) & (data["TIME"] <= last_)]
        self.TEST_DATA = data[data["TIME"] == last_+1]
        if self.TRAINING_DATA.shape[0] == 0 or self.TEST_DATA.shape[0] == 0:
            raise Exception("Training data hoặc test data đang rỗng.")

        self.PROFIT = np.array(self.TRAINING_DATA["PROFIT"], dtype=np.float64)
        self.TEST_PROFIT = np.array(self.TEST_DATA["PROFIT"], dtype=np.float64)

        self.OPERAND = np.transpose(np.array(self.TRAINING_DATA.drop(columns=drop_cols), dtype=np.float64))
        self.TEST_OPERAND = np.transpose(np.array(self.TEST_DATA.drop(columns=drop_cols), dtype=np.float64))

        time_arr = np.array(self.TRAINING_DATA["TIME"])
        qArr = np.unique(time_arr)
        self.INDEX = np.full(qArr.shape[0]+1, 0, dtype=np.int64)
        for i in range(qArr.shape[0]):
            if i == qArr.shape[0] - 1:
                self.INDEX[qArr.shape[0]] = time_arr.shape[0]
            else:
                temp = time_arr[self.INDEX[i]]
                for j in range(self.INDEX[i], time_arr.shape[0]):
                    if time_arr[j] != temp:
                        self.INDEX[i+1] = j
                        break

        # Profit method
        self.profit_method = profit_method
        self.profit_method_index = ["geomean", "harmean", "bitmean"].index(profit_method)
        self.get_profit_by_weight = getattr(nopy, f"get_{profit_method}_profit_by_weight")


    def convert_formula_to_str(self, formula):
        temp = "+-*/"
        str_formula = ""
        for i in range(formula.shape[0]):
            if i % 2 == 1:
                str_formula += str(formula[i])
            else:
                str_formula += temp[formula[i]]

        return str_formula


    def convert_str_to_formula(self, str_formula):
        temp = "+-*/"
        f_len = sum(str_formula.count(c) for c in temp) * 2
        str_len = len(str_formula)
        if self.OPERAND.shape[0] <= 256:
            formula = np.full(f_len, 0, dtype=np.uint8)
        else:
            formula = np.full(f_len, 0, dtype=np.uint16)

        idx = 0
        for i in range(f_len):
            if i % 2 == 1:
                t_ = 0
                while True:
                    t_ = 10*t_ + int(str_formula[idx])
                    idx += 1
                    if idx == str_len or str_formula[idx] in temp:
                        break

                formula[i] = t_
            else:
                formula[i] = temp.index(str_formula[idx])
                idx += 1

        return formula


    def get_formula_profit(self, formula):
        if type(formula) == str:
            formula = self.convert_str_to_formula(formula)

        weight = nopy.calculate_formula(formula, self.OPERAND)
        return self.get_profit_by_weight(weight, self.PROFIT, self.INDEX)


    def explain_formula_old(self, formula):
        if type(formula) == str:
            formula = self.convert_str_to_formula(formula)

        weight = nopy.calculate_formula(formula, self.OPERAND)

        min_time = self.TRAINING_DATA.iloc[-1]["TIME"]
        Out_Time = np.arange(min_time, min_time+self.INDEX.shape[0]-1)
        Out_Profit = []
        Out_Symbol = []
        Out_Value = []
        for i in range(self.INDEX.shape[0]-2, -1, -1):
            temp = weight[self.INDEX[i]:self.INDEX[i+1]]
            max_temp = np.max(temp)
            max_ = np.where(temp == max_temp)[0] + self.INDEX[i]
            if max_.shape[0] == 1:
                Out_Profit.append(self.PROFIT[max_[0]])
                Out_Symbol.append(self.TRAINING_DATA.iloc[max_[0]]["SYMBOL"])
                Out_Value.append(max_temp)
            else:
                Out_Profit.append(1.0)
                Out_Symbol.append('NI')
                Out_Value.append(max_temp)

        return pd.DataFrame({
            'Out_Time': Out_Time,
            'Out_Symbol': Out_Symbol,
            'Out_Profit': Out_Profit,
            'Out_Value': Out_Value
        })

    def explain_formula(self, formula, not_invest_available = False):
        if type(formula) == str:
            formula = self.convert_str_to_formula(formula)

        weight = nopy.calculate_formula(formula, self.OPERAND)

        min_time = self.TRAINING_DATA.iloc[-1]["TIME"]
        Out_Time = np.arange(min_time, min_time+self.INDEX.shape[0]-1)
        Out_Profit = []
        Out_Symbol = []
        Out_Value = []
        list_rank_not_invest = []       #Hiếu thêm
        for i in range(self.INDEX.shape[0]-2, -1, -1):
            temp = weight[self.INDEX[i]:self.INDEX[i+1]]
            ##
            if not_invest_available:
                id_not_invest = np.where(self.TRAINING_DATA["SYMBOL"][self.INDEX[i]:self.INDEX[i+1]] == 'NOT_INVEST')[0]       #Hiếu thêm
                rank_not_invest = len(np.where(temp > temp[id_not_invest])[0])                                                  #Hiếu thêm
                list_rank_not_invest.append(rank_not_invest)                                                                    #Hiếu thêm
            max_temp = np.max(temp)
            max_ = np.where(temp == max_temp)[0] + self.INDEX[i]
            if max_.shape[0] == 1:
                Out_Profit.append(self.PROFIT[max_[0]])
                Out_Symbol.append(self.TRAINING_DATA.iloc[max_[0]]["SYMBOL"])
                Out_Value.append(max_temp)
            else:
                Out_Profit.append(1.0)
                Out_Symbol.append('NI')
                Out_Value.append(max_temp)

        #Hiếu thêm
        if not_invest_available:
            return pd.DataFrame({
                    'Out_Time': Out_Time,
                    'Out_Symbol': Out_Symbol,
                    'Out_Profit': Out_Profit,
                    'Out_Value': Out_Value,
                    'Out_Rank_NI': list_rank_not_invest
                })
        else:
            return pd.DataFrame({
                'Out_Time': Out_Time,
                'Out_Symbol': Out_Symbol,
                'Out_Profit': Out_Profit,
                'Out_Value': Out_Value
            })


    def get_invested_company(self, formula):
        if type(formula) == str:
            formula = self.convert_str_to_formula(formula)

        weight = nopy.calculate_formula(formula, self.TEST_OPERAND)
        max_weight = np.max(weight)
        max_ = np.where(weight == max_weight)[0]
        if max_.shape[0] == 1:
            com = self.TEST_DATA.iloc[max_[0]]["SYMBOL"]
            prof = self.TEST_PROFIT[max_[0]]
            val = max_weight
        else:
            com = "NI"
            prof = 1.0
            val = max_weight

        return com, prof, val


    def convert_npy_file_to_DataFrame(self, path_or_2d_formula_array):
        if type(path_or_2d_formula_array) == str:
            list_formula = np.load(path_or_2d_formula_array, allow_pickle=True)
        else:
            list_formula = path_or_2d_formula_array

        list_str_formula = []
        list_profit = []
        list_next_invest = []
        list_next_profit = []
        for i in range(list_formula.shape[0]):
            formula = list_formula[i]
            list_str_formula.append(self.convert_formula_to_str(formula))
            list_profit.append(self.get_formula_profit(formula))
            com, prof, val = self.get_invested_company(formula)
            list_next_invest.append(com)
            list_next_profit.append(prof)

        return pd.DataFrame({
            "formula": list_str_formula,
            self.profit_method+"_profit": list_profit,
            "invest": list_next_invest,
            "profit": list_next_profit
        })


    def find_threshold(self, formula, target):
        if type(formula) == str:
            formula = self.convert_str_to_formula(formula)

        return nopy.find_max_threshold(formula, self.OPERAND, self.INDEX, self.PROFIT, target, self.profit_method_index)
    

    def get_value_invest_threshold(self, formula, threshold):
        if type(formula) == str:
            formula = self.convert_str_to_formula(formula)

        a, b, c = nopy.get_value_invest_threshold(formula, threshold, self.TEST_OPERAND, self.TEST_PROFIT)
        if b == -1:
            return a, "BANK", c
        elif b == -2:
            return a, "NI", c
        else:
            return a, self.TEST_DATA.iloc[b]["SYMBOL"], c
            

    def value_limit_filter(self, df:pd.DataFrame, target_profit):
        """
        Đầu vào:
            * dataframe có cột formula
            * Những công thức có profit nhỏ hơn target, profit_limit sẽ được đặt thành 0.
        """
        key1 = self.profit_method + "_profit"
        key2_ = key1[0:3] + "_limit"
        key2 = key2_[0].upper() + key2_[1:]

        data = df[["formula"]]
        data[key1] = np.full(data.shape[0], -1.0)
        data["Value_limit"] = np.full(data.shape[0], -1.7976931348623157e+308)
        data[key2] = np.full(data.shape[0], -1.0)
        data["Time_invest"] = np.full(data.shape[0], self.TEST_DATA.iloc[0]["TIME"])
        data["Value_invest"] = np.full(data.shape[0], 0.0)
        data["Com_invest"] = np.full(data.shape[0], "NI")
        data["Profit_invest"] = np.full(data.shape[0], 0.0)
        for i in range(data.shape[0]):
            data[key1][i], data["Value_limit"][i], data[key2][i] = self.find_threshold(data["formula"][i], target_profit)
            if data[key2][i] > 0.0:
                data["Value_invest"][i], data["Com_invest"][i], data["Profit_invest"][i] = self.get_value_invest_threshold(data["formula"][i], data["Value_limit"][i])

        return data[data[key2] > 0.0]

        