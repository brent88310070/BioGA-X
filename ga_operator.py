import numpy as np
import random
from deap import creator

# -----------------------------------------------
# ------------------ Mutation -------------------
# -----------------------------------------------
def mutation_xAI_guided(individual, saliency, alphabet,
                        base_prob=0.1, factor=3, exclude_first_pos=False):
    """
    xAI-guided mutation
    """

    seq = list(individual)

    start = 1 if exclude_first_pos and len(seq) > 0 else 0
    body_seq = seq[start:]

    L = len(body_seq)
    if L == 0:
        return creator.Individual(seq)

    sal = np.asarray(saliency)
    sal = sal[start:start + L]

    if len(sal) < L:
        L = len(sal)
        body_seq = body_seq[:L]

    min_val = sal.min()
    max_val = sal.max()
    if max_val - min_val < 1e-6:
        norm_sal = np.zeros_like(sal, dtype=float)
    else:
        norm_sal = (sal - min_val) / (max_val - min_val)

    mutation_probs = base_prob * factor * (1.0 - norm_sal)
    for i in range(L):
        if random.random() < mutation_probs[i]:
            pos = i + start
            old = seq[pos]
            choices = [aa for aa in alphabet if aa != old]
            if choices:
                seq[pos] = random.choice(choices)

    return creator.Individual(seq)



def mutation_standard(individual, alphabet, indpb, exclude_first_pos = False):
    """
    標準隨機突變 (Uniform Mutation)，但可以鎖定第一個位置。
    """
    seq = list(individual)
    start_idx = 1 if exclude_first_pos else 0
    
    for i in range(start_idx, len(seq)):
        if random.random() < indpb:
            choices = [aa for aa in alphabet if aa != seq[i]]
            seq[i] = random.choice(choices)
            
    return creator.Individual(seq)

# -----------------------------------------------
# ------------------ Crossover ------------------
# -----------------------------------------------
def smooth_array(arr, window_size):
    """計算移動視窗總和"""
    kernel = np.ones(window_size)
    return np.convolve(arr, kernel, mode='valid')

def crossover_xAI_guided(ind1, ind2, saliency1, saliency2, min_k=3, max_k=8, exclude_first_pos=False, first_word="M"):
    """
    雙向 xAI Crossover：
    雙方都嘗試用對方的強項來修補自己的弱項。
    """

    def repair_individual(receiver_ind, donor_ind, receiver_sal, donor_sal, start_offset):
        """
        嘗試將 donor 的高分片段植入 receiver 的低分片段
        回傳: (是否成功, 新序列)
        """
        # 1. 計算 Saliency 差異 (Donor - Receiver)
        # 我們希望找到 donor 比 receiver 強很多的地方
        diff_map = donor_sal - receiver_sal
        
        best_gain = -float('inf') # 差異越大越好
        best_start_idx = -1
        best_len = -1
        
        # 限制視窗搜尋範圍
        valid_len = len(diff_map)
        real_max_k = min(valid_len, max_k)
        
        if valid_len < min_k:
            return False, receiver_ind # 太短無法操作

        # 動態視窗搜尋最佳替換點 (Find Max Gain)
        for k in range(min_k, real_max_k + 1):
            # 這裡計算的是「差異的總和」，也就是替換後能提升多少分數
            gain_arr = smooth_array(diff_map, k)
            
            # 找最大增益 (Max Gain)
            max_val = np.max(gain_arr)
            idx = np.argmax(gain_arr)
            
            if max_val > best_gain:
                best_gain = max_val
                best_start_idx = idx
                best_len = k
        
        # 只有當增益大於 0 (Donor 確實比較好) 時才交換
        if best_gain > 0:
            cut_start = start_offset + best_start_idx
            cut_end = cut_start + best_len
            
            new_seq = list(receiver_ind[:cut_start]) + \
                      list(donor_ind[cut_start:cut_end]) + \
                      list(receiver_ind[cut_end:])
            if exclude_first_pos and len(new_seq) > 0:
                new_seq[0] = first_word
                
            return True, creator.Individual(new_seq)
            
        return False, receiver_ind

    # --- 主流程 ---

    start_index = 1 if exclude_first_pos else 0
    
    sal1_proc = saliency1[start_index:]
    sal2_proc = saliency2[start_index:]
    
    # 雙向修復 (Bidirectional Repair)
    # Child 1: 以 Ind1 為本體，嘗試用 Ind2 修補
    success1, child1 = repair_individual(ind1, ind2, sal1_proc, sal2_proc, start_index)
    # Child 2: 以 Ind2 為本體，嘗試用 Ind1 修補
    success2, child2 = repair_individual(ind2, ind1, sal2_proc, sal1_proc, start_index)

    # 3. 多樣性策略：如果都沒有發生交換，強制進行隨機交換 (選用)
    if not success1 and not success2:
        if len(ind1) > min_k + 1:
            cx_point = random.randint(1, len(ind1)-1)
            c1_seq = list(ind1[:cx_point]) + list(ind2[cx_point:])
            c2_seq = list(ind2[:cx_point]) + list(ind1[cx_point:])
            
            if exclude_first_pos:
                c1_seq[0] = first_word
                c2_seq[0] = first_word
                
            child1 = creator.Individual(c1_seq)
            child2 = creator.Individual(c2_seq)

    return child1, child2


def cxTwoPoint_standard(ind1, ind2, min_k=3, max_k=8, exclude_first_pos=False):
    """
    Two-point crossover with controlled segment length [min_k, max_k]
    """
    L = len(ind1)
    start = 1 if exclude_first_pos else 0
    max_k = min(max_k, L - start)

    if max_k < min_k:
        return ind1, ind2

    seg_len = random.randint(min_k, max_k)
    i = random.randint(start, L - seg_len)
    j = i + seg_len

    # swap segment
    ind1[i:j], ind2[i:j] = ind2[i:j], ind1[i:j]
    return ind1, ind2