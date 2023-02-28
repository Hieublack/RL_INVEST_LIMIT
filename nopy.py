from numba import njit
import numpy as np
""""""

@njit
def calculate_formula(formula, operand):
    temp_0 = np.zeros(operand.shape[1])
    temp_1 = temp_0.copy()
    temp_op = -1
    for i in range(1, formula.shape[0], 2):
        if formula[i] >= operand.shape[0]:
            raise

        if formula[i-1] < 2:
            temp_op = formula[i-1]
            temp_1 = operand[formula[i]].copy()
        else:
            if formula[i-1] == 2:
                temp_1 *= operand[formula[i]]
            else:
                temp_1 /= operand[formula[i]]

        if i+1 == formula.shape[0] or formula[i+1] < 2:
            if temp_op == 0:
                temp_0 += temp_1
            else:
                temp_0 -= temp_1

    temp_0[np.isnan(temp_0)] = -1.7976931348623157e+308
    temp_0[np.isinf(temp_0)] = -1.7976931348623157e+308
    return temp_0


@njit
def get_geomean_profit_by_weight(weight, profit, index):
    temp_profit = 1.0
    for i in range(index.shape[0]-2, -1, -1):
        temp = weight[index[i]:index[i+1]]
        max_ = np.where(temp == np.max(temp))[0] + index[i]
        if max_.shape[0] == 1:
            if profit[max_[0]] <= 0.0:
                return 0.0

            temp_profit *= profit[max_[0]]

    return temp_profit**(1.0/(index.shape[0]-1))


@njit
def get_harmean_profit_by_weight(weight, profit, index):
    denominator = 0.0
    for i in range(index.shape[0]-2, -1, -1):
        temp = weight[index[i]:index[i+1]]
        max_ = np.where(temp == np.max(temp))[0] + index[i]
        if max_.shape[0] == 1:
            if profit[max_[0]] <= 0.0:
                return 0.0

            denominator += 1.0/profit[max_[0]]
        else:
            denominator += 1.0

    return (index.shape[0]-1)/denominator


@njit
def get_bitmean_profit_by_weight(weight, profit, index):
    com_los = np.where(profit < 1.0)[0]
    com_win = np.where(profit > 1.0)[0]
    max_los = np.max(weight[com_los])
    a = np.count_nonzero(weight[com_win] < max_los)
    return 1 - (a/com_win.shape[0])


@njit
def get_valid_operand(formula, struct, idx, start, num_operand):
    valid_operand = np.full(num_operand, 0)
    valid_operand[start:num_operand] = 1

    for i in range(struct.shape[0]):
        if struct[i,2] + 2*struct[i,1] > idx:
            gr_idx = i
            break

    """
    Tránh hoán vị nhân chia trong một cụm
    """
    pre_op = formula[idx-1]
    if pre_op >= 2:
        if pre_op == 2:
            temp_idx = struct[gr_idx,2]
            if idx >= temp_idx + 2:
                valid_operand[0:formula[idx-2]] = 0
        else:
            temp_idx = struct[gr_idx,2]
            temp_idx_1 = temp_idx + 2*struct[gr_idx,3]
            if idx > temp_idx_1 + 2:
                valid_operand[0:formula[idx-2]] = 0

            """
            Tránh chia lại những toán hạng đã nhân ở trong cụm (chỉ phép chia mới check)
            """
            valid_operand[formula[temp_idx:temp_idx_1+1:2]] = 0

    """
    Tránh hoán vị cộng trừ các cụm, kể từ cụm thứ 2 trở đi
    """
    if gr_idx > 0:
        gr_check_idx = -1
        for i in range(gr_idx-1,-1,-1):
            if struct[i,0]==struct[gr_idx,0] and struct[i,1]==struct[gr_idx,1] and struct[i,3]==struct[gr_idx,3]:
                gr_check_idx = i
                break

        if gr_check_idx != -1:
            idx_ = 0
            while True:
                idx_1 = struct[gr_idx,2] + idx_
                idx_2 = struct[gr_check_idx,2] + idx_
                if idx_1 == idx:
                    valid_operand[0:formula[idx_2]] = 0
                    break

                if formula[idx_1] != formula[idx_2]:
                    break

                idx_ += 2

        """
        Tránh trừ đi những cụm đã cộng trước đó (chỉ ở trong trừ cụm mới check)
        """
        if struct[gr_idx,0] == 1 and idx + 2 == struct[gr_idx,2] + 2*struct[gr_idx,1]:
            list_gr_check = np.where((struct[:,0]==0) & (struct[:,1]==struct[gr_idx,1]) & (struct[:,3]==struct[gr_idx,3]))[0]
            for i in list_gr_check:
                temp_idx = struct[i,2] + 2*struct[i,1] - 2
                temp_idx_1 = struct[gr_idx,2] + 2*struct[gr_idx,1] - 2
                if (formula[struct[i,2]:temp_idx] == formula[struct[gr_idx,2]:temp_idx_1]).all():
                    valid_operand[formula[temp_idx]] = 0

    return np.where(valid_operand==1)[0]


@njit
def get_profitsss_by_weightsss(weights, profit, index, profit_method):
    result = np.zeros(weights.shape[0])
    for i in range(weights.shape[0]):
        result[i] = profit_method(weights[i], profit, index)

    return result


@njit
def split_posint_into_sum(n, arr, list_result):
    if np.sum(arr) == n:
        list_result.append(arr)
    else:
        idx = np.where(arr==0)[0][0]
        sum_ = np.sum(arr)
        if idx == 0:
            max_ = n
        else:
            max_ = arr[idx-1]

        max_ = min(n-sum_, max_)
        for i in range(max_, 0, -1):
            arr[idx] = i
            split_posint_into_sum(n, arr.copy(), list_result)


@njit
def create_struct(add_struct, sub_struct):
    struct = np.full((add_struct.shape[0]+sub_struct.shape[0], 4), -1)
    temp_val = 1
    for i in range(add_struct.shape[0]):
        struct[i,:] = np.array([0, add_struct[i], temp_val, add_struct[i]-1])
        temp_val += 2*struct[i,1]

    for i in range(sub_struct.shape[0]):
        temp_val_1 = add_struct.shape[0] + i
        struct[temp_val_1,:] = np.array([1, sub_struct[i], temp_val, sub_struct[i]-1])
        temp_val += 2*struct[temp_val_1,1]

    return struct


@njit
def create_formula(struct):
    n = np.sum(struct[:,1])
    formula = np.full(2*n, 0)
    temp_val = 0
    for i in range(struct.shape[0]):
        temp = struct[i]
        formula[temp_val] = temp[0]
        temp_val += 2
        for j in range(temp[1]-1):
            if j < temp[3]:
                formula[temp_val] = 2
            else:
                formula[temp_val] = 3

            temp_val += 2

    return formula


@njit
def update_struct(struct, numerator_condition):
    if numerator_condition:
        for i in range(struct.shape[0]-1, -1, -1):
            if struct[i,3] > (struct[i,1]-1)//2:
                temp = np.where((struct[i:,0]==struct[i,0]) & (struct[i:,1]==struct[i,1]))[0] + i
                struct[temp,3] = struct[i,3] - 1
                temp_1 = np.max(temp) + 1
                struct[temp_1:,3] = struct[temp_1:,1] - 1
                return True

        return False
    else:
        for i in range(struct.shape[0]-1, -1, -1):
            if struct[i,3] > 0:
                temp = np.where((struct[i:,0]==struct[i,0]) & (struct[i:,1]==struct[i,1]))[0] + i
                struct[temp,3] = struct[i,3] - 1
                temp_1 = np.max(temp) + 1
                struct[temp_1:,3] = struct[temp_1:,1] - 1
                return True

        return False


@njit
def sub_get_valid_idxsss_and_targetsss(weight, profit, index, num_test, profit_method_index):
    overall_profit = np.zeros(num_test)
    num_test_1 = num_test - 1
    if profit_method_index == 0:
        temp_profit = 1.0
        for i in range(index.shape[0]-2, -1, -1):
            temp = weight[index[i]:index[i+1]]
            max_ = np.where(temp == np.max(temp))[0] + index[i]
            if max_.shape[0] == 1:
                if profit[max_[0]] <= 0.0:
                    if i <= num_test_1:
                        overall_profit[num_test_1-i:] = 0.0
                    else:
                        overall_profit[0:] = 0.0
                    return overall_profit

                temp_profit *= profit[max_[0]]

            if i <= num_test_1:
                overall_profit[num_test_1-i] = temp_profit

        return overall_profit

    elif profit_method_index == 1:
        temp_deno = 0.0
        for i in range(index.shape[0]-2, -1, -1):
            temp = weight[index[i]:index[i+1]]
            max_ = np.where(temp == np.max(temp))[0] + index[i]
            if max_.shape[0] == 1:
                if profit[max_[0]] <= 0.0:
                    if i <= num_test_1:
                        overall_profit[num_test_1-i:] = 1.7976931348623157e+308
                    else:
                        overall_profit[0:] = 1.7976931348623157e+308
                    return overall_profit

                temp_deno += 1.0/profit[max_[0]]
            else:
                temp_deno += 1.0

            if i <= num_test_1:
                overall_profit[num_test_1-i] = temp_deno

        return overall_profit
    
    elif profit_method_index == 2:
        for i in range(num_test_1, -1, -1):
            temp_weight_arr = weight[index[i]:index[-1]]
            temp_profit_arr = profit[index[i]:index[-1]]
            com_los = np.where(temp_profit_arr < 1.0)[0]
            com_win = np.where(temp_profit_arr > 1.0)[0]
            max_los = np.max(temp_weight_arr[com_los])
            a = np.count_nonzero(temp_weight_arr[com_win] < max_los)
            overall_profit[num_test_1-i] = 1 - (a/com_win.shape[0])

        return overall_profit


@njit
def get_valid_idxsss_and_targetsss(weights, profit, index, num_test, target, profit_method_index):
    two_d_profits = np.zeros((weights.shape[0], num_test))
    for i in range(weights.shape[0]):
        two_d_profits[i] = sub_get_valid_idxsss_and_targetsss(weights[i], profit, index, num_test, profit_method_index)

    test_start = index.shape[0] - num_test
    if profit_method_index == 0:
        for i in range(num_test):
            two_d_profits[:,i] **= (1.0/(test_start+i))

    elif profit_method_index == 1:
        for i in range(num_test):
            two_d_profits[:,i] = (test_start+i) / two_d_profits[:,i]
    
    elif profit_method_index == 2:
        pass

    check_target = np.where(two_d_profits >= target, 1, 0)
    check_valid = np.full(weights.shape[0], 0)
    for i in range(weights.shape[0]):
        if (check_target[i]==1).any():
            check_valid[i] = 1

    temp = np.where(check_valid==1)[0]

    return temp, check_target[temp]


@njit
def get_valid_op(formula, struct, idx, start):
    valid_op = np.full(2, 0)
    valid_op[start-2:] = 1

    if idx // 2 <= struct[0,1] // 2:
        valid_op[1] = 0

    return np.where(valid_op == 1)[0] + 2


@njit
def get_threshold(weight, index, profit, profit_method_index):
    thresholds = np.zeros(index.shape[0]-1, dtype=np.float64)
    profits_nguong = np.zeros(index.shape[0]-1)
    if profit_method_index == 0: # geomean
        temp_profit = 1.0
        for i in range(index.shape[0]-2, -1, -1):
            temp = weight[index[i]:index[i+1]]
            max_temp = np.max(temp)
            max_ = np.where(temp == max_temp)[0] + index[i]
            if max_.shape[0] == 1:
                if profit[max_[0]] == 0.0:
                    temp_profit = 0.0
                else:
                    temp_profit *= profit[max_[0]]
                thresholds[index.shape[0]-2-i] = max_temp
                profits_nguong[index.shape[0]-2-i] = profit[max_[0]]
            else:
                thresholds[index.shape[0]-2-i] = 1.7976931348623157e+308
                profits_nguong[index.shape[0]-2-i] = 1.0

        min_ = np.min(thresholds)
        x_nguong = min_ - np.max(np.array([1e-9, 1e-9*np.abs(min_)]))
        return temp_profit**(1.0/(index.shape[0]-1)), thresholds, profits_nguong, x_nguong

    elif profit_method_index == 1: # harmean
        temp_denomirator = 0.0
        for i in range(index.shape[0]-2, -1, -1):
            temp = weight[index[i]:index[i+1]]
            max_temp = np.max(temp)
            max_ = np.where(temp == max_temp)[0] + index[i]
            if max_.shape[0] == 1:
                if profit[max_[0]] == 0.0:
                    temp_denomirator = 1.7976931348623157e+308
                else:
                    temp_denomirator += 1.0/profit[max_[0]]

                thresholds[index.shape[0]-2-i] = max_temp
                profits_nguong[index.shape[0]-2-i] = profit[max_[0]]
            else:
                temp_denomirator += 1.0
                thresholds[index.shape[0]-2-i] = 1.7976931348623157e+308
                profits_nguong[index.shape[0]-2-i] = 1.0

        min_ = np.min(thresholds)
        x_nguong = min_ - np.max(np.array([1e-9, 1e-9*np.abs(min_)]))
        return (index.shape[0]-1)/temp_denomirator, thresholds, profits_nguong, x_nguong
    
    elif profit_method_index == 2: # bitmean
        raise


@njit
def find_max_threshold(formula, operand, index, profit, target, profit_method_index):
    weight = calculate_formula(formula, operand)
    abcxyz_mean_profit, thresholds, profits_nguong, x_nguong = get_threshold(weight, index, profit, profit_method_index)
    if abcxyz_mean_profit < target:
        return abcxyz_mean_profit, 0.0, 0.0

    max_profit = abcxyz_mean_profit
    if profit_method_index == 0: # geomean
        for x in thresholds:
            temp_profit = 1.0
            for i in range(thresholds.shape[0]):
                if thresholds[i] > x:
                    if profits_nguong[i] == 0.0:
                        temp_profit = 0.0
                        break
                    else:
                        temp_profit *= profits_nguong[i]
                else:
                    temp_profit *= 1.01

            geo_ = temp_profit**(1.0/(index.shape[0]-1))
            if geo_ > max_profit:
                max_profit = geo_
                x_nguong = x

        return abcxyz_mean_profit, x_nguong, max_profit

    if profit_method_index == 1: # harmean
        for x in thresholds:
            temp_denomirator = 0.0
            for i in range(thresholds.shape[0]):
                if thresholds[i] > x:
                    if profits_nguong[i] == 0.0:
                        temp_denomirator = 1.7976931348623157e+308
                        break
                    else:
                        temp_denomirator += 1.0/profits_nguong[i]
                else:
                    temp_denomirator += 1.0/1.01

            har_ = (index.shape[0]-1)/temp_denomirator
            if har_ > max_profit:
                max_profit = har_
                x_nguong = x

        return abcxyz_mean_profit, x_nguong, max_profit
    
    if profit_method_index == 2: # bitmean
        raise


@njit
def get_value_invest_threshold(formula, threshold, test_operand, test_profit):
    weight = calculate_formula(formula, test_operand)
    max_value = np.max(weight)
    # if max_value <= threshold:
    #     return max_value, -1, 1.01
    # else:
    max_ = np.where(weight == max_value)[0]
    if max_.shape[0] == 1:
        return max_value, max_[0], test_profit[max_[0]]
    else:
        return max_value, -2, 1.0
        