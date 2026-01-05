import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from deap import base, creator, tools
from Bio import Align

from multi_class_cnn import load_model, evaluate_population, compute_captum_ig_batch
from ga_operator import mutation_xAI_guided, mutation_standard, crossover_xAI_guided, cxTwoPoint_standard

# ==========================================
# 1. 全域參數設定 (Global Configuration)
# ==========================================

# Cluster 3 seq (Initial seq)
INPUT_PROT = "MAGRAIFSVSCSSTPSLCIPYSTASFSSMNRLALPAVRISPRTNRFPRIHCSMSANDIKAGTNIEVDGAP"

# Cluster 1
TARGET_CLASS = 1 

POP_SIZE  = 100           # 最終族群數量
N_GEN = 200               # 最大世代數
IG_STEPS = 30             # Captum IG 步數
GROWTH_THRESHOLD = 0.001  # 成長率門檻
NO_GROWTH_LIMIT = 5       # 連續沒成長就停
BASE_MUT_PROB = 0.05      # Mutation prob
SEQ_LEN = 70              # 假設序列長度固定為 70
NUM_CLASSES = 5           # 多分類模型
SEQ_TYPE = "PROTEIN"      # DNA / PROTEIN

MODEL_SAVE_PATH = "best_cv_models.pth" 
RESULTS_DIR = "ga_results_figures"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [f'C{i}' for i in range(0, NUM_CLASSES)]

# ==========================================
# 2. 類別定義 (Classes)
# ==========================================

class SequenceConfig:
    def __init__(self, seq_type="PROTEIN"):
        if seq_type.upper() == "DNA":
            self.alphabet = list("ACGT")
        elif seq_type.upper() == "PROTEIN":
            self.alphabet = list("ACDEFGHIKLMNPQRSTVWY")
        else:
            raise ValueError("Type must be 'DNA' or 'PROTEIN'")
        self.seq_type = seq_type

# ==========================================
# 3. Initial population Operators
# ==========================================

def enforce_start_aa(seq, start_aa="M"):
    s = list(seq)
    if start_aa is not None:
        s[0] = start_aa
    return s

def mutate_substitutions(seq, alphabet, min_k=1, max_k=5, protect_first=True):
    s = list(seq)
    L = len(s)
    start = 1 if protect_first else 0
    valid_positions = list(range(start, L))
    if not valid_positions: return s
    k = random.randint(min_k, max_k)
    positions = random.sample(valid_positions, k=min(k, len(valid_positions)))
    for pos in positions:
        aa_old = s[pos]
        choices = [aa for aa in alphabet if aa != aa_old]
        if choices: s[pos] = random.choice(choices)
    return s

def insertion_indel(seq, alphabet, min_k=1, max_k=5, protect_first=True):
    s = list(seq)
    L = len(s)
    start = 1 if protect_first else 0
    k = random.randint(min_k, max_k)
    for _ in range(k):
        pos = random.randrange(start, L + 1)
        s.insert(pos, random.choice(alphabet))
        s = s[:L]
    return s

def deletion_indel(seq, alphabet, min_k=1, max_k=5, protect_first=True):
    s = list(seq)
    L = len(s)
    if L <= 10: return s
    start = 1 if protect_first else 0
    valid_positions = list(range(start, L))
    if not valid_positions: return s
    k = random.randint(min_k, max_k)
    for _ in range(k):
        pos = random.choice(valid_positions)
        del s[pos]
        s.append(random.choice(alphabet))
    return s

def shuffle_segment(seq, min_len=3, max_len=8, protect_first=True):
    s = list(seq)
    L = len(s)
    start = 1 if protect_first else 0
    if L - start < min_len: return s
    seg_len = random.randint(min_len, min(max_len, L - start))
    i = random.randint(start, L - seg_len)
    j = i + seg_len - 1
    seg = s[i:j+1]
    random.shuffle(seg)
    s[i:j+1] = seg
    return s

def reverse_segment(seq, min_len=3, max_len=8, protect_first=True):
    s = list(seq)
    L = len(s)
    start = 1 if protect_first else 0
    max_len = min(max_len, L - start)
    if max_len < min_len: return s
    seg_len = random.randint(min_len, max_len)
    i = random.randint(start, L - seg_len)
    j = i + seg_len - 1
    s[i:j+1] = s[i:j+1][::-1]
    return s

# ==========================================
# 4. GA 初始化設定 (Setup)
# ==========================================

def setup_ga_initialization(seed_sequence, seq_type="PROTEIN", pop_size=100, protect_start=False):
    config = SequenceConfig(seq_type)
    alphabet = config.alphabet

    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    op_map = {
        "mutation": lambda s: mutate_substitutions(s, alphabet, protect_first=protect_start),
        "insertion": lambda s: insertion_indel(s, alphabet, protect_first=protect_start),
        "deletion": lambda s: deletion_indel(s, alphabet, protect_first=protect_start),
        "shuffle": shuffle_segment,
        "reverse": reverse_segment,
    }
    op_names = list(op_map.keys())
    op_weights = [1, 1, 1, 1, 1]

    def init_individual_generator():
        op = random.choices(op_names, weights=op_weights, k=1)[0]
        seq_list = op_map[op](seed_sequence)
        if protect_start: seq_list = enforce_start_aa(seq_list, start_aa="M")
        return creator.Individual(seq_list)

    toolbox.register("individual", init_individual_generator)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(n=pop_size)
    return pop, toolbox, config

# ==========================================
# 5. 評估與相似度函數 (Evaluation & Similarity)
# ==========================================

def needleman_wunsch_similarity(seq, ref, match_score=1.0, mismatch_score=-1.0, gap_open=-10, gap_extend=-0.5):
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_score
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend
    alignments = aligner.align(seq, ref)
    best_alignment = alignments[0]
    aln_len = best_alignment.shape[1] 
    matches = best_alignment.counts().identities
    return (matches / aln_len)

def evaluate_population_multi(population, model, target_cluster, ref_seq):
    cnn_scores = evaluate_population(population, model, target_cluster, seq_len=SEQ_LEN, 
                                     batch_size=128, device=DEVICE)
    for ind, cnn_fit in zip(population, cnn_scores):
        cnn_score = float(cnn_fit[0])
        seq = "".join(ind)
        sim = needleman_wunsch_similarity(seq, ref_seq)
        ind.fitness.values = (cnn_score, sim)

# ==========================================
# 6. GA 主迴圈 (Main Loop)
# ==========================================

def select_right_upper_topk(fits, w=0.5, k=20):
    cnn, sim = fits[:, 0], fits[:, 1]
    score = w * cnn + (1 - w) * sim
    topk_idx = np.argsort(score)[-k:]
    return np.mean(cnn[topk_idx]), np.mean(sim[topk_idx])

def run_ga_nsga2_xai(pop, toolbox, model, device, target_class, alphabet, ref_seq, n_gen=N_GEN, start=None, xAI_guided_p=0.5):
    pop_size = len(pop)
    history = {"gen": [], "mean_cnn": [], "mean_sim": [], "ru_cnn": [], "ru_sim": []}

    invalid = [x for x in pop if not x.fitness.valid]
    if invalid: evaluate_population_multi(invalid, model, target_class, ref_seq)
    pop[:] = tools.selNSGA2(pop, pop_size)

    best_overall, best_score = None, -1.0
    best_front = None
    prev_cnn = prev_sim = None 
    no_growth = 0
    exclude_first_pos = False if start is None else True

    for gen in range(1, n_gen + 1):
        print(f"\n===== Gen {gen} =====")
        parents = [toolbox.clone(x) for x in tools.selTournamentDCD(pop, pop_size)]
        sal = compute_captum_ig_batch(parents, model, target_cluster=target_class, device=device, seq_len=SEQ_LEN, n_steps=IG_STEPS)

        children = []
        for i in range(0, pop_size, 2):
            p1, s1 = parents[i], sal[i]
            p2, s2 = (parents[i+1], sal[i+1]) if i+1 < pop_size else (toolbox.clone(p1), s1)

            if random.random() < xAI_guided_p:
                c1, c2 = crossover_xAI_guided(p1, p2, s1, s2, exclude_first_pos=exclude_first_pos, first_word=start)
            else:
                c1, c2 = cxTwoPoint_standard(toolbox.clone(p1), toolbox.clone(p2), exclude_first_pos=exclude_first_pos)
                if start != None: c1[0] = c2[0] = start

            def mutate(c, s):
                return mutation_xAI_guided(c, s, alphabet, BASE_MUT_PROB, exclude_first_pos=exclude_first_pos) \
                    if random.random() < xAI_guided_p else \
                    mutation_standard(c, alphabet, BASE_MUT_PROB, exclude_first_pos=exclude_first_pos)

            c1, c2 = [mutate(c, s) for c, s in ((c1, s1), (c2, s2))]
            if hasattr(c1.fitness, "values"): del c1.fitness.values
            if hasattr(c2.fitness, "values"): del c2.fitness.values
            children += [c1, c2]

        children = children[:pop_size]
        invalid = [x for x in children if not x.fitness.valid]
        if invalid: evaluate_population_multi(invalid, model, target_class, ref_seq)
        pop[:] = tools.selNSGA2(pop + children, pop_size)

        fits = np.array([x.fitness.values for x in pop], float)
        cnn, sim = fits[:,0], fits[:,1]
        mean_cnn, mean_sim = cnn.mean(), sim.mean()
        history["gen"].append(gen)
        history["mean_cnn"].append(float(mean_cnn))
        history["mean_sim"].append(float(mean_sim))
        print(f"  CNN mean={mean_cnn:.4f} | SIM mean={mean_sim:.4f}")

        ru_cnn, ru_sim = select_right_upper_topk(fits, w=0.5, k=20)
        history["ru_cnn"].append(float(ru_cnn))
        history["ru_sim"].append(float(ru_sim))
        print(f"  Top 20 CNN mean={ru_cnn:.4f} | Top 20 SIM mean={ru_sim:.4f}")

        bi = cnn.argmax()
        if cnn[bi] > best_score:
            best_score = cnn[bi]
            best_overall = toolbox.clone(pop[bi])

        front = tools.sortNondominated(pop, len(pop), True)[0]
        if best_front is None or len(front) > len(best_front):
            best_front = [toolbox.clone(x) for x in front]

        if prev_cnn is not None:
            if (ru_cnn - prev_cnn < GROWTH_THRESHOLD) and (ru_sim - prev_sim < GROWTH_THRESHOLD):
                no_growth += 1
                print(f" Top-K no-growth {no_growth}/{NO_GROWTH_LIMIT}")
            else:
                no_growth = 0
        prev_cnn, prev_sim = ru_cnn, ru_sim

        if no_growth >= NO_GROWTH_LIMIT:
            print("\n Early stopping triggered based on Top-K stagnation\n")
            break

    return pop, history, best_overall, best_front

# ==========================================
# 7. 繪圖函數 (Plotting) - 修改版
# ==========================================

def plot_ga_trajectory_nsga2_rightup(history, save_path=None):
    g = history["gen"]
    plt.figure(figsize=(8,5))
    plt.plot(g, history["ru_cnn"], label="Model score")
    plt.plot(g, history["ru_sim"], label="Seq similarity")
    plt.xlabel("Generation"); plt.ylabel("Objective value")
    plt.title("NSGA-II Evolution (CNN vs Similarity)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # <--- 新增：如果有指定路徑，就存檔
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Trajectory plot saved to {save_path}")
        
    plt.show()

def select_right_upper_topk_from_fits(fits, w=0.5, k=20):
    cnn, sim = fits[:, 0], fits[:, 1]
    score = w * cnn + (1 - w) * sim
    topk_idx = np.argsort(score)[-k:]
    return topk_idx

# <--- 修改：新增 save_path 參數
def plot_final_pareto_scatter(final_pop, w=0.5, k=20, save_path=None):
    fits = np.array([ind.fitness.values for ind in final_pop], dtype=float)
    cnn_scores = fits[:, 0]
    sim_scores = fits[:, 1]

    ru_idx = select_right_upper_topk_from_fits(fits, w=w, k=k)

    plt.figure(figsize=(7, 6))
    plt.scatter(sim_scores, cnn_scores, s=40, alpha=0.6, edgecolor="k", label="Population")

    plt.scatter(
        sim_scores[ru_idx],
        cnn_scores[ru_idx],
        s=40,
        c="red",
        edgecolor="k",
        label=f"Right-upper (Top-{k})"
    )

    plt.xlabel("Similarity (Needleman–Wunsch)")
    plt.ylabel("CNN score (target probability)")
    plt.title("Final Population – Pareto Scatter\n(CNN vs Similarity)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    # <--- 新增：如果有指定路徑，就存檔
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Pareto plot saved to {save_path}")

    plt.show()

# ==========================================
# 8. 主程式執行區塊 (Main Execution)
# ==========================================

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Target Class: {TARGET_CLASS}")
    
    # <--- 新增：建立儲存圖片的資料夾 (如果不存在的話)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved in: {RESULTS_DIR}")

    # 1. 載入模型
    try:
        model = load_model(MODEL_SAVE_PATH, DEVICE, NUM_CLASSES, SEQ_LEN)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'best_cv_models.pth' exists or update MODEL_SAVE_PATH.")
        exit()

    # 2. 初始 fitness
    pop_prot, toolbox, config = setup_ga_initialization(
        seed_sequence=INPUT_PROT, seq_type=SEQ_TYPE, pop_size=POP_SIZE, protect_start=True
    )
    evaluate_population_multi(pop_prot, model, TARGET_CLASS, INPUT_PROT)

    # 3. 執行 NSGA-II + XAI
    final_pop, history, best_ind_overall, best_pareto_front = run_ga_nsga2_xai(
        pop_prot, toolbox, model, DEVICE, TARGET_CLASS, config.alphabet, INPUT_PROT,
        n_gen=N_GEN, start="M", xAI_guided_p=0.5
    )

    # 4. 繪製並儲存結果
    print("Plotting and saving results...")
    
    # <--- 修改：傳入 save_path
    traj_path = os.path.join(RESULTS_DIR, "evolution_trajectory.png")
    plot_ga_trajectory_nsga2_rightup(history, save_path=traj_path)
    
    pareto_path = os.path.join(RESULTS_DIR, "pareto_front.png")
    plot_final_pareto_scatter(final_pop, w=0.5, k=20, save_path=pareto_path)
    
    print("Done.")