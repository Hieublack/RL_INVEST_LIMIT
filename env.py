from index import*



@njit()
def initEnv_drop():
    env_state = np.zeros(ENV_LENGTH)

    env_state[ENV_PLAYER_CAN_INVEST : ENV_PLAYER_CAN_INVEST + NUMBER_PLAYER] = 1

    ids = np.random.choice(ALL_INDEX, 2, replace= False)
    env_state[ENV_ALL_IN4_FIRST : ENV_ALL_IN4_SECOND] = np.concatenate((ALL_PROFIT[ids[0]], ALL_VALUE[ids[0]], ALL_RANK_NOT_INVEST[ids[0]], ALL_LIMIT[ids[0]]))
    env_state[ENV_ALL_IN4_SECOND : ] = np.concatenate((ALL_PROFIT[ids[1]], ALL_VALUE[ids[1]], ALL_RANK_NOT_INVEST[ids[1]], ALL_LIMIT[ids[1]]))
    env_state[ENV_CURRENT_QUARTER] = NUMBER_HISTORY

    env_state[ENV_FIRST_HISTORY : ENV_FIRST_GMEAN] = env_state[ENV_ALL_IN4_FIRST : ENV_ALL_IN4_FIRST + NUMBER_QUARTER][: NUMBER_HISTORY]
    env_state[ENV_FIRST_GMEAN] = gmean( env_state[ENV_ALL_IN4_FIRST : ENV_ALL_IN4_FIRST + NUMBER_QUARTER][: int(env_state[ENV_CURRENT_QUARTER] - QUARTER_PER_CYCLE)])
    env_state[ENV_FIRST_HMEAN] = hmean( env_state[ENV_ALL_IN4_FIRST : ENV_ALL_IN4_FIRST + NUMBER_QUARTER][: int(env_state[ENV_CURRENT_QUARTER] - QUARTER_PER_CYCLE)])
    env_state[ENV_FIRST_RANK_NOT_INVEST] = env_state[int(ENV_ALL_IN4_FIRST + NUMBER_QUARTER * 2 + env_state[ENV_CURRENT_QUARTER])]
    env_state[ENV_FIRST_LIMIT] = env_state[int(ENV_ALL_IN4_FIRST + NUMBER_QUARTER * 3 + env_state[ENV_CURRENT_QUARTER])]
    env_state[ENV_FIRST_VALUE] = env_state[int(ENV_ALL_IN4_FIRST + NUMBER_QUARTER * 1 + env_state[ENV_CURRENT_QUARTER])]
    
    env_state[ENV_SECOND_HISTORY : ENV_SECOND_GMEAN] = env_state[ENV_ALL_IN4_SECOND : ENV_ALL_IN4_SECOND + NUMBER_QUARTER][: NUMBER_HISTORY]
    env_state[ENV_SECOND_GMEAN] = gmean( env_state[ENV_ALL_IN4_SECOND : ENV_ALL_IN4_SECOND + NUMBER_QUARTER][: int(env_state[ENV_CURRENT_QUARTER] - QUARTER_PER_CYCLE)])
    env_state[ENV_SECOND_HMEAN] = hmean( env_state[ENV_ALL_IN4_SECOND : ENV_ALL_IN4_SECOND + NUMBER_QUARTER][: int(env_state[ENV_CURRENT_QUARTER] - QUARTER_PER_CYCLE)])
    env_state[ENV_SECOND_RANK_NOT_INVEST] = env_state[int(ENV_ALL_IN4_SECOND + NUMBER_QUARTER * 2 + env_state[ENV_CURRENT_QUARTER])]
    env_state[ENV_SECOND_LIMIT] = env_state[int(ENV_ALL_IN4_SECOND + NUMBER_QUARTER * 3 + env_state[ENV_CURRENT_QUARTER])]
    env_state[ENV_SECOND_VALUE] = env_state[int(ENV_ALL_IN4_SECOND + NUMBER_QUARTER * 1 + env_state[ENV_CURRENT_QUARTER])]
    

    return env_state


@njit()
def initEnv():
    env_state = np.zeros(ENV_LENGTH)
    env_state[ENV_PLAYER_CAN_INVEST : ENV_PLAYER_CAN_INVEST + NUMBER_PLAYER] = 1
    env_state[ENV_PROFIT_AGENT : ENV_PROFIT_AGENT + NUMBER_PLAYER] = 1


    # ids = np.random.choice(ALL_INDEX, 2, replace= False)
    ids = np.array([0,1])
    env_state[ENV_ALL_IN4_FIRST : ENV_ALL_IN4_SECOND] = np.concatenate((ALL_PROFIT[ids[0]], ALL_VALUE[ids[0]], ALL_RANK_NOT_INVEST[ids[0]], ALL_LIMIT[ids[0]]))
    env_state[ENV_ALL_IN4_SECOND : ] = np.concatenate((ALL_PROFIT[ids[1]], ALL_VALUE[ids[1]], ALL_RANK_NOT_INVEST[ids[1]], ALL_LIMIT[ids[1]]))
    #thời điểm bắt đầu
    env_state[ENV_CURRENT_QUARTER] = START_QUARTER

    #lịch sử profit là tính đến 4 quý trước quý hiện tại
    env_state[ENV_FIRST_HISTORY : ENV_FIRST_GMEAN] = env_state[ENV_ALL_IN4_FIRST : ENV_ALL_IN4_FIRST + NUMBER_QUARTER][: NUMBER_HISTORY]
    env_state[ENV_FIRST_GMEAN] = gmean( env_state[ENV_ALL_IN4_FIRST : ENV_ALL_IN4_FIRST + NUMBER_QUARTER][: NUMBER_HISTORY])
    env_state[ENV_FIRST_HMEAN] = hmean( env_state[ENV_ALL_IN4_FIRST : ENV_ALL_IN4_FIRST + NUMBER_QUARTER][: NUMBER_HISTORY])
    env_state[ENV_FIRST_RANK_NOT_INVEST] = env_state[int(ENV_ALL_IN4_FIRST + NUMBER_QUARTER * 2 + env_state[ENV_CURRENT_QUARTER])]
    env_state[ENV_FIRST_LIMIT] = env_state[int(ENV_ALL_IN4_FIRST + NUMBER_QUARTER * 3 + env_state[ENV_CURRENT_QUARTER])]
    env_state[ENV_FIRST_VALUE] = env_state[int(ENV_ALL_IN4_FIRST + NUMBER_QUARTER * 1 + env_state[ENV_CURRENT_QUARTER])]
    
    env_state[ENV_SECOND_HISTORY : ENV_SECOND_GMEAN] = env_state[ENV_ALL_IN4_SECOND : ENV_ALL_IN4_SECOND + NUMBER_QUARTER][: NUMBER_HISTORY]
    env_state[ENV_SECOND_GMEAN] = gmean( env_state[ENV_ALL_IN4_SECOND : ENV_ALL_IN4_SECOND + NUMBER_QUARTER][: NUMBER_HISTORY])
    env_state[ENV_SECOND_HMEAN] = hmean( env_state[ENV_ALL_IN4_SECOND : ENV_ALL_IN4_SECOND + NUMBER_QUARTER][: NUMBER_HISTORY])
    env_state[ENV_SECOND_RANK_NOT_INVEST] = env_state[int(ENV_ALL_IN4_SECOND + NUMBER_QUARTER * 2 + env_state[ENV_CURRENT_QUARTER])]
    env_state[ENV_SECOND_LIMIT] = env_state[int(ENV_ALL_IN4_SECOND + NUMBER_QUARTER * 3 + env_state[ENV_CURRENT_QUARTER])]
    env_state[ENV_SECOND_VALUE] = env_state[int(ENV_ALL_IN4_SECOND + NUMBER_QUARTER * 1 + env_state[ENV_CURRENT_QUARTER])]

    return env_state

@njit()
def getAgentState(env_state):
    id_action = int(env_state[ENV_ID_ACTION])
    player_state = np.zeros(P_LENGTH)
    player_state[: P_PROFIT_MULTI] = env_state[: ENV_PLAYER_CAN_INVEST]
    player_state[P_PROFIT_MULTI : P_PROFIT_MULTI + NUMBER_PLAYER] = np.concatenate((env_state[ENV_PROFIT_AGENT + id_action : ENV_PROFIT_AGENT + NUMBER_PLAYER], env_state[ENV_PROFIT_AGENT : ENV_PROFIT_AGENT+ id_action ]))
    player_state[P_CHECK_END] = env_state[ENV_CHECK_END]
    player_state[P_PLAYER_CAN_INVEST] = env_state[ENV_PLAYER_CAN_INVEST + id_action]

    player_state[P_UPDATE_RESULT : P_UPDATE_RESULT + NUMBER_PLAYER] = np.concatenate((env_state[ENV_UPDATE_RESULT + id_action : ENV_UPDATE_RESULT + NUMBER_PLAYER], env_state[ENV_UPDATE_RESULT : ENV_UPDATE_RESULT + id_action]))

    return player_state

@njit()
def getReward(player_state):
    value_return = -1
    if player_state[P_CHECK_END] == 0:
        return value_return
    else:
        result = player_state[P_PROFIT_MULTI : P_PROFIT_MULTI + NUMBER_PLAYER]
        max_ = np.max(result)
        if np.argmax(player_state[P_PROFIT_MULTI : P_PROFIT_MULTI + NUMBER_PLAYER]) != 0:
            return 0
        else:
            if len(np.where(result == max_)[0]) > 1:
                return 0
            else:
                return 1

@njit()
def check_winner(env_state):
    result = env_state[ENV_PROFIT_AGENT : ENV_PROFIT_AGENT + NUMBER_PLAYER]
    # print('Tích profit: ', result)
    winner = np.argmax(result)
    if np.max(result) == 0:
        winner = -1
    return winner


@njit()
def getValidActions(player_state):
    list_action_return = np.zeros(AMOUNT_ACTION)
    if player_state[P_PLAYER_CAN_INVEST] == 1:
        list_action_return[:] = 1
    else:
        list_action_return[0] = 1
    return list_action_return

@njit()
def getActionSize():
    return 3

@njit()
def getAgentSize():
    return 2

@njit()
def getStateSize():
    return 48


@njit()
def stepEnv(env_state, action):
    id_action = int(env_state[ENV_ID_ACTION])
    #cập nhật giá trị người chơi
    if env_state[ENV_PLAYER_CAN_INVEST + id_action] == 1:
        #xử lí action 
        if action != 0:
            if action == 1:
                env_state[ENV_PROFIT_DELAY_AGENT + id_action] = env_state[ENV_ALL_IN4_FIRST + int(env_state[ENV_CURRENT_QUARTER])]
            elif action == 2:
                env_state[ENV_PROFIT_DELAY_AGENT + id_action] = env_state[ENV_ALL_IN4_SECOND + int(env_state[ENV_CURRENT_QUARTER])]
            #cập nhật trạng thái đầu tư của người chơi
            env_state[ENV_PLAYER_CAN_INVEST + id_action] = 0
            env_state[ENV_UPDATE_RESULT + id_action] = 0
            env_state[ENV_COUNT_DELAY_AGENT + id_action] = 0

    else:
        #tương đương là người chơi action 0 
        #cập nhật đếm số quý agent chờ
        env_state[ENV_COUNT_DELAY_AGENT + id_action] += 1
    
    #kiểm tra xem tất cả người chơi action hết chưa để nhảy quý, nếu action hết rồi thì cập nhật lợi nhuận nếu có, cập nhật quý mới và các thông tin về profit, ngưỡng, value
    env_state[ENV_COUNT_PLAYER_ACTION] += 1   
    if env_state[ENV_COUNT_PLAYER_ACTION] == NUMBER_PLAYER:
        #kiểm tra xem có ai đang chờ update lợi nhuận ko:
        profit_delay_agent = env_state[ENV_PROFIT_DELAY_AGENT : ENV_PROFIT_DELAY_AGENT + NUMBER_PLAYER].copy()
        done_delay = env_state[ENV_COUNT_DELAY_AGENT : ENV_COUNT_DELAY_AGENT + NUMBER_PLAYER].copy()
        update_result = env_state[ENV_UPDATE_RESULT : ENV_UPDATE_RESULT + NUMBER_PLAYER].copy()
        player_can_invest = env_state[ENV_PLAYER_CAN_INVEST : ENV_PLAYER_CAN_INVEST + NUMBER_PLAYER].copy()
        # print('check delay', profit_delay_agent, done_delay, update_result, player_can_invest)

        player_need_update = np.where(done_delay == 3)[0]

        update_result[player_need_update] = 1
        player_can_invest[player_need_update] = 1

        profit_use_update = np.ones(NUMBER_PLAYER)
        profit_use_update[player_need_update] = profit_delay_agent[player_need_update]

        profit_delay_agent[player_need_update] = 1

        #cập nhật tích lợi nhuận
        env_state[ENV_PROFIT_AGENT : ENV_PROFIT_AGENT + NUMBER_PLAYER] *= profit_use_update
        #cập nhật trạng thái cập nhật
        env_state[ENV_UPDATE_RESULT : ENV_UPDATE_RESULT + NUMBER_PLAYER] = update_result
        #cập nhật profit delay
        env_state[ENV_PROFIT_DELAY_AGENT : ENV_PROFIT_DELAY_AGENT + NUMBER_PLAYER] = profit_delay_agent
        #cập nhật trạng thái khả năng đầu tư
        env_state[ENV_PLAYER_CAN_INVEST : ENV_PLAYER_CAN_INVEST + NUMBER_PLAYER] = player_can_invest

        #cập nhật quý mới và các thông tin về profit, ngưỡng, value, history, gmean, hmean, ranknotinvest
        env_state[ENV_CURRENT_QUARTER] += 1
        curent_quarter_history = int(env_state[ENV_CURRENT_QUARTER]) - QUARTER_PER_CYCLE

        #lịch sử profit là tính đến 4 quý trước quý hiện tại
        env_state[ENV_FIRST_HISTORY : ENV_FIRST_GMEAN] = env_state[ENV_ALL_IN4_FIRST : ENV_ALL_IN4_FIRST + NUMBER_QUARTER][curent_quarter_history - NUMBER_HISTORY: curent_quarter_history]
        env_state[ENV_FIRST_GMEAN] = gmean( env_state[ENV_ALL_IN4_FIRST : ENV_ALL_IN4_FIRST + NUMBER_QUARTER][: curent_quarter_history])
        env_state[ENV_FIRST_HMEAN] = hmean( env_state[ENV_ALL_IN4_FIRST : ENV_ALL_IN4_FIRST + NUMBER_QUARTER][: curent_quarter_history])
        env_state[ENV_FIRST_RANK_NOT_INVEST] = env_state[int(ENV_ALL_IN4_FIRST + NUMBER_QUARTER * 2 + env_state[ENV_CURRENT_QUARTER])]
        env_state[ENV_FIRST_LIMIT] = env_state[int(ENV_ALL_IN4_FIRST + NUMBER_QUARTER * 3 + env_state[ENV_CURRENT_QUARTER])]
        env_state[ENV_FIRST_VALUE] = env_state[int(ENV_ALL_IN4_FIRST + NUMBER_QUARTER * 1 + env_state[ENV_CURRENT_QUARTER])]
        
        env_state[ENV_SECOND_HISTORY : ENV_SECOND_GMEAN] = env_state[ENV_ALL_IN4_SECOND : ENV_ALL_IN4_SECOND + NUMBER_QUARTER][curent_quarter_history - NUMBER_HISTORY: curent_quarter_history]
        env_state[ENV_SECOND_GMEAN] = gmean( env_state[ENV_ALL_IN4_SECOND : ENV_ALL_IN4_SECOND + NUMBER_QUARTER][: curent_quarter_history])
        env_state[ENV_SECOND_HMEAN] = hmean( env_state[ENV_ALL_IN4_SECOND : ENV_ALL_IN4_SECOND + NUMBER_QUARTER][: curent_quarter_history])
        env_state[ENV_SECOND_RANK_NOT_INVEST] = env_state[int(ENV_ALL_IN4_SECOND + NUMBER_QUARTER * 2 + env_state[ENV_CURRENT_QUARTER])]
        env_state[ENV_SECOND_LIMIT] = env_state[int(ENV_ALL_IN4_SECOND + NUMBER_QUARTER * 3 + env_state[ENV_CURRENT_QUARTER])]
        env_state[ENV_SECOND_VALUE] = env_state[int(ENV_ALL_IN4_SECOND + NUMBER_QUARTER * 1 + env_state[ENV_CURRENT_QUARTER])]
        #reset lại đếm người chơi quý mới
        env_state[ENV_COUNT_PLAYER_ACTION] = 0
    #chuyển người chơi
    env_state[ENV_ID_ACTION] = int(env_state[ENV_ID_ACTION] + 1) % NUMBER_PLAYER

    return env_state


@njit()
def system_check_end(env_state):
    if env_state[ENV_CURRENT_QUARTER] >= NUMBER_QUARTER:
        return False
    return True

def one_game(list_player, per_file):
    env_state = initEnv()
    count_turn = 0
    while system_check_end(env_state) and count_turn < 2000:
        p_id_action = int(env_state[ENV_ID_ACTION])
        p_state = getAgentState(env_state)
        action, per_file = list_player[p_id_action](p_state, per_file)
        list_action = getValidActions(p_state)
        if list_action[action] != 1:
            raise Exception("action ko hợp lệ")
        # print(f'ở quý {int(env_state[ENV_CURRENT_QUARTER])} Player {p_id_action} có list_action {list_action}, thực hiện action {action}' )
        # print(f'Kết quả hiện tại: {env_state[ENV_PROFIT_AGENT : ENV_PROFIT_AGENT + NUMBER_PLAYER]}')
        
        env_state = stepEnv(env_state, action)
        count_turn += 1

    env_state[ENV_CHECK_END] = 1
    winner = check_winner(env_state)                        
    for id_player in range(NUMBER_PLAYER):
        env_state[ENV_ID_ACTION] = id_player
        p_state = getAgentState(env_state)
        action, per_file = list_player[id_player](p_state, per_file)
    
    return winner, per_file

def normal_main(list_player, times, file_per):
    count = np.zeros(len(list_player)+1)
    all_id_player = np.arange(len(list_player))
    for van in range(times):
        shuffle = np.random.choice(all_id_player, NUMBER_PLAYER, replace=False)
        shuffle_player = [list_player[shuffle[0]], list_player[shuffle[1]]]
        winner, file_per = one_game(shuffle_player, file_per)
        if winner == -1:
            count[winner] += 1
        else:
            count[shuffle[winner]] += 1
    return list(count.astype(np.int64)), file_per



@njit()
def numba_one_game(p_lst_idx_shuffle, p0, p1, per_file):
    env_state = initEnv()
    count_turn = 0
    while system_check_end(env_state):
        p_idx = int(env_state[ENV_ID_ACTION])
        p_state = getAgentState(env_state)
        if p_lst_idx_shuffle[p_idx] == 0:
            act, per_file = p0(p_state, per_file)
        elif p_lst_idx_shuffle[p_idx] == 1:
            act, per_file = p1(p_state, per_file)
        
        if getValidActions(p_state)[act] != 1:
            raise Exception('bot dua ra action khong hop le')
        env_state = stepEnv(env_state, act)
        count_turn += 1

    env_state[ENV_CHECK_END] = 1
    winner = check_winner(env_state)
    for id_player in range(NUMBER_PLAYER):
        p_state = getAgentState(env_state)
        p_idx = int(env_state[ENV_ID_ACTION])
        if p_lst_idx_shuffle[p_idx] == 0:
            act, per_file = p0(p_state, per_file)
        elif p_lst_idx_shuffle[p_idx] == 1:
            act, per_file = p1(p_state, per_file)
    
        env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % NUMBER_PLAYER
    return winner, per_file

@njit()
def numba_main(p0, p1, num_game,per_file):
    count = np.zeros(getAgentSize()+1, dtype= np.int32)
    p_lst_idx = np.array([0,1])
    for _n in range(num_game):
        np.random.shuffle(p_lst_idx)
        winner, per_file = numba_one_game(p_lst_idx, p0, p1, per_file )
        if winner == -1:
            count[winner] += 1
        else:
            count[p_lst_idx[winner]] += 1
    return count, per_file

@jit()
def one_game_numba(p0, list_other, per_player, per1, p1):
    env_state = initEnv()
    count_turn = 0
    while system_check_end(env_state):
        idx = int(env_state[ENV_ID_ACTION])
        player_state = getAgentState(env_state)
        if list_other[idx] == -1:
            action, per_player = p0(player_state,per_player)
        elif list_other[idx] == 1:
            action, per1 = p1(player_state,per1)

        if getValidActions(player_state)[action] != 1:
            raise Exception('bot dua ra action khong hop le')

        env_state = stepEnv(env_state, action)
        count_turn += 1

    env_state[ENV_CHECK_END] = 1

    for p_idx in range(NUMBER_PLAYER):
        if list_other[int(env_state[ENV_ID_ACTION])] == -1:
            act, per_player = p0(getAgentState(env_state), per_player)
        env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 5
    
    winner = False
    if np.where(list_other == -1)[0] ==  check_winner(env_state): 
        winner = True
    else: 
        winner = False

    return winner,  per_player



@jit()
def n_game_numba(p0, num_game, per_player, list_other, per1, p1):
    win = 0
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba(p0, list_other, per_player, per1, p1)
        win += winner
    return win, per_player


@njit()
def random_Env(p_state, per):
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], per

# import importlib.util, json, sys
# from setup import SHOT_PATH

# def load_module_player(player):
#     spec = importlib.util.spec_from_file_location('Agent_player', f"{SHOT_PATH}Agent/{player}/Agent_player.py")
#     module = importlib.util.module_from_spec(spec)
#     sys.modules[spec.name] = module 
#     spec.loader.exec_module(module)
#     return module


def numba_main_2(p0, n_game, per_player, level, *args):
    list_other = np.array([1, -1])
    if level == 0:
        per_agent_env = np.array([0])
        return n_game_numba(p0, n_game, per_player, list_other, per_agent_env, random_Env )
    else:
        env_name = sys.argv[1]
        if len(args) > 0:
            dict_level = json.load(open(f'{SHOT_PATH}Log/check_system_about_level.json'))
        else:
            dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))

        if str(level) not in dict_level[env_name]:
            raise Exception('Hiện tại không có level này') 
        lst_agent_level = dict_level[env_name][str(level)][2]

        p1 = load_module_player(lst_agent_level[0]).Test
        # p2 = load_module_player(lst_agent_level[1]).Test
        # p3 = load_module_player(lst_agent_level[2]).Test
        # p4 = load_module_player(lst_agent_level[3]).Test
        per_level = []
        for id in range(getAgentSize()-1):
            data_agent_env = list(np.load(f'{SHOT_PATH}Agent/{lst_agent_level[id]}/Data/{env_name}_{level}/Train.npy',allow_pickle=True))
            per_level.append(data_agent_env)
        
        return n_game_numba(p0, n_game, per_player, list_other, per_level[0], per_level[1], p1)











