from base import Method
import pandas as pd
import numpy as np
import os
from numba import njit
from numba.typed import List
from colorama import Fore, Style
import copy
from datetime import datetime
import nopy

import warnings
warnings.filterwarnings("ignore")


class CompleteMethod(Method):
    def __init__(self, data: pd.DataFrame, path_save: str, num_training: int, profit_method: str) -> None:
        super().__init__(data, path_save, num_training, profit_method)


    def fill_operand(self, formula, struct, idx, temp_0, temp_op, temp_1, target):
        start = -1
        if (formula[0:idx]==self.current[5][0:idx]).all():
            start = self.current[5][idx]
        else:
            start = 0

        valid_operand = nopy.get_valid_operand(formula, struct, idx, start, self.OPERAND.shape[0])
        if valid_operand.shape[0] > 0:
            if formula[idx-1] < 2:
                temp_op_new = formula[idx-1]
                temp_1_new = self.OPERAND[valid_operand].copy()
            else:
                temp_op_new = temp_op
                if formula[idx-1] == 2:
                    temp_1_new = temp_1 * self.OPERAND[valid_operand]
                else:
                    temp_1_new = temp_1 / self.OPERAND[valid_operand]

            if idx + 1 == formula.shape[0] or formula[idx+1] < 2:
                if temp_op_new == 0:
                    temp_0_new = temp_0 + temp_1_new
                else:
                    temp_0_new = temp_0 - temp_1_new
            else:
                temp_0_new = np.array([temp_0]*valid_operand.shape[0])

            if idx + 1 == formula.shape[0]:
                temp_0_new[np.isnan(temp_0_new)] = -1.7976931348623157e+308
                temp_0_new[np.isinf(temp_0_new)] = -1.7976931348623157e+308
                temp_profits = nopy.get_profitsss_by_weightsss(temp_0_new, self.PROFIT, self.INDEX, self.get_profit_by_weight)
                valid_formula = np.where(temp_profits>=target)[0]
                if valid_formula.shape[0] > 0:
                    temp_list_formula = np.array([formula]*valid_formula.shape[0])
                    temp_list_formula[:,idx] = valid_operand[valid_formula]
                    self.list_formula[self.count[0]:self.count[0]+valid_formula.shape[0]] = temp_list_formula
                    self.count[0:3:2] += valid_formula.shape[0]

                self.current[5][:] = formula[:]
                self.current[5][idx] = self.OPERAND.shape[0]

                if self.count[0] >= self.count[1] or self.count[2] >= self.count[3]:
                    return True
            else:
                temp_list_formula = np.array([formula]*valid_operand.shape[0])
                temp_list_formula[:,idx] = valid_operand
                idx_new = idx + 2
                for i in range(valid_operand.shape[0]):
                    if self.fill_operand(temp_list_formula[i], struct, idx_new, temp_0_new[i], temp_op_new, temp_1_new[i], target):
                        return True

        return False


    def generate_formula(self, target_profit=1.0, formula_file_size=1000000, target_num_formula=1000000000, numerator_condition=False):
        '''
        * target_profit: Lợi nhuận mong muốn.
        * formula_file_size: Số lượng công thức xấp xỉ trong mỗi file lưu trữ (nên để từ 2 triệu đổ xuống, tránh tràn RAM).
        * target_num_formula: Số công thức đạt điều kiện được sinh trong 1 lần chạy ko ngắt.
        * numerator_condition: Nếu được truyền vào là True, các công thức sinh ra có số phần tử trên tử số lớn hơn hoặc bằng số phần tử dưới mẫu số.
        '''
        print(Fore.LIGHTYELLOW_EX+"Khi ngắt bằng tay thì cần tự chạy phương thức <CompleteMethod_object>.save_history() để lưu lịch sử.", Style.RESET_ALL)

        try:
            temp = np.load(self.path+"history.npy", allow_pickle=True)
            self.history = temp
        except:
            self.history = [
                1, # Số toán hạng có trong công thức
                0, # Số toán hạng trong các trừ cụm
                0, # Cấu trúc các cộng cụm thứ mấy
                0, # Cấu trúc các trừ cụm thứ mấy
                np.array([[0, 1, 1, 0]]), # Cấu trúc công thức tổng quát
                np.array([0, 0]) # Công thức đã sinh đến trong lịch sử
            ]

        self.current = copy.deepcopy(self.history)

        self.count = np.array([0, formula_file_size, 0, target_num_formula])

        num_operand = self.history[0] - 1
        while True:
            num_operand += 1
            print("Đang chạy sinh công thức có số toán hạng là ", num_operand, ". . .")
            if self.OPERAND.shape[0] <= 256:
                self.list_formula = np.full((formula_file_size+self.OPERAND.shape[0], 2*num_operand), 0, dtype=np.uint8)
            else:
                self.list_formula = np.full((formula_file_size+self.OPERAND.shape[0], 2*num_operand), 0, dtype=np.uint16)

            if num_operand == self.history[0]:
                start_num_sub_operand = self.history[1]
            else: start_num_sub_operand = 0

            for num_sub_operand in range(start_num_sub_operand, num_operand+1):
                temp_arr = np.full(num_sub_operand, 0)
                list_sub_struct = List([temp_arr])
                list_sub_struct.pop(0)
                nopy.split_posint_into_sum(num_sub_operand, temp_arr, list_sub_struct)

                num_add_operand = num_operand - num_sub_operand
                temp_arr = np.full(num_add_operand, 0)
                list_add_struct = List([temp_arr])
                list_add_struct.pop(0)
                nopy.split_posint_into_sum(num_add_operand, temp_arr, list_add_struct)

                if num_sub_operand == self.history[1] and num_operand == self.history[0]:
                    start_add_struct_idx = self.history[2]
                else: start_add_struct_idx = 0

                for add_struct_idx in range(start_add_struct_idx, len(list_add_struct)):
                    if  add_struct_idx == self.history[2] and \
                        num_sub_operand == self.history[1] and num_operand == self.history[0]:
                        start_sub_struct_idx = self.history[3]
                    else: start_sub_struct_idx =  0

                    for sub_struct_idx in range(start_sub_struct_idx, len(list_sub_struct)):
                        add_struct = list_add_struct[add_struct_idx][list_add_struct[add_struct_idx]>0]
                        sub_struct = list_sub_struct[sub_struct_idx][list_sub_struct[sub_struct_idx]>0]
                        if  sub_struct_idx == self.history[3] and add_struct_idx == self.history[2] and \
                            num_sub_operand == self.history[1] and num_operand == self.history[0]:
                            struct = self.history[4].copy()
                        else: struct = nopy.create_struct(add_struct, sub_struct)

                        while True:
                            if struct.shape == self.history[4].shape and (struct==self.history[4]).all():
                                formula = self.history[5].copy()
                            else:
                                formula = nopy.create_formula(struct)

                            self.current[0] = num_operand
                            self.current[1] = num_sub_operand
                            self.current[2] = add_struct_idx
                            self.current[3] = sub_struct_idx
                            self.current[4] = struct.copy()
                            self.current[5] = formula.copy()

                            while self.fill_operand(formula, struct, 1, np.zeros(self.OPERAND.shape[1]), -1, np.zeros(self.OPERAND.shape[1]), target_profit):
                                self.save_history()

                            if not nopy.update_struct(struct, numerator_condition):
                                break

            if self.save_history():
                break

        return


    def save_history(self):
        '''
        Lưu lịch sử: trong trường hợp ngắt bằng tay.
        '''
        np.save(self.path+"history.npy", self.current)
        print(Fore.LIGHTGREEN_EX+"Đã lưu lịch sử.", Style.RESET_ALL)
        if self.count[0] == 0:
            return False

        num_operand = self.current[0]
        while True:
            pathSave = self.path + f"high_profit_{num_operand}_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".npy"
            if not os.path.exists(pathSave):
                np.save(pathSave, self.list_formula[0:self.count[0]])
                self.count[0] = 0
                print(Fore.LIGHTGREEN_EX+"Đã lưu công thức", Style.RESET_ALL)
                if self.count[2] >= self.count[3]:
                    raise Exception("Đã sinh đủ công thức theo yêu cầu.")

                return False

