import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients


AA_LIST = list('AGILMPVFWYCNSTQDEHKR')
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

def seq_to_onehot(seq, max_len=70):
    onehot = np.zeros((20, max_len), dtype=np.float32)
    for i, char in enumerate(seq[:max_len]):
        if char in AA_TO_IDX:
            onehot[AA_TO_IDX[char], i] = 1.0
    return onehot


class CNNModel(nn.Module):
    def __init__(self, num_classes=5, input_channels=20, seq_len=70):
        super(CNNModel, self).__init__()
        
        # 特徵提取層
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )

        self._to_linear = None
        self._calculate_flatten_size(input_channels, seq_len)
        
        # 分類層
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Dropout(0.3), # 建議在 FC 層間也加 Dropout 或是保持原樣
            nn.Linear(128, num_classes)
        )

    def _calculate_flatten_size(self, input_channels, seq_len):
        with torch.no_grad():
            x = torch.zeros(1, input_channels, seq_len)
            x = self.features(x)
            self._to_linear = x.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x
    

def evaluate_population(population, model, target_cluster=3, seq_len=70, batch_size=128, device="cuda"):
    model.eval()

    onehot_list = []
    for ind in population:
        seq = "".join(ind)
        onehot_list.append(seq_to_onehot(seq, max_len=seq_len))  # numpy (20, L)

    X = np.stack(onehot_list)  # shape (B, 20, L)

    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, target_cluster]
            scores.extend(probs.cpu().numpy())

    return [(float(s),) for s in scores]


def compute_captum_ig_batch(population, model, target_cluster, device="cuda", seq_len=70, n_steps=10):
    """
    使用 Captum 的 Integrated Gradients 計算 Saliency Map。
    
    Args:
        population: List of individuals (sequences)
        model: Trained CNN model
        target_cluster: 目標類別 index
        n_steps: IG 的積分步數
    
    Returns:
        saliency_map: numpy array (Batch, Length)
    """
    model.eval()
    model.zero_grad()

    onehot_list = [seq_to_onehot("".join(ind), max_len=seq_len) for ind in population]
    input_tensor = torch.tensor(np.stack(onehot_list), dtype=torch.float32).to(device)

    # Baseline (全零)
    baseline_tensor = torch.zeros_like(input_tensor).to(device)

    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(
        inputs=input_tensor,
        baselines=baseline_tensor,
        target=target_cluster,
        n_steps=n_steps,
        return_convergence_delta=True,
        internal_batch_size=32 
    )
    
    # Aggregation
    saliency_map = attributions.sum(dim=1)
    return saliency_map.detach().cpu().numpy()


def load_model(MODEL_SAVE_PATH, DEVICE, NUM_CLASSES, SEQ_LEN):
    model = CNNModel(num_classes=NUM_CLASSES, seq_len=SEQ_LEN)
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["best_model"])
    return model.to(DEVICE)