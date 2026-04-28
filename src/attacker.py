import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod

# ==============================================
# Base Class for Attacker Module
# ==============================================
class BaseAttacker(ABC):
    """
    Abstract Base Class for all attack algorithm aiming time series 
    """
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        
    def _validate_input(self, x):
        """input dim safety checker"""
        assert isinstance(x, torch.Tensor), "input 'x' must be a Pytorch Tensor"
        assert x.ndim == 3, f"Input 'x' must be 3D (Batch, Length Feature), got {x.shape}"
        
    @abstractmethod
    def attack(self, x, y_hat=None):
        """
        return: perturbation matrix
        """
        pass

# ==============================================
# Subclass realizing attack algorithm
# ==============================================
class GWNAttacker(BaseAttacker):
    """
    Guassian White Noise - using for Baseline
    """
    def __init__(self, model, loss_fn, scale=0.05):
        super().__init__(model, loss_fn)
        self.scale = scale
    
    def attack(self, x, y_hat=None):
        self._validate_input(x)
        w = torch.randn_like(x) * self.scale
        return w
    
class TSAttacker(BaseAttacker):
    """
    Time Sparse Attack: a black box attack using Subaspace Pursuit
    """
    def __init__(self, model, loss_fn, tau, epsilon, max_iter, is_channel_independent=False):
        super().__init__(model, loss_fn)
        self.tau = tau
        self.epsilon = epsilon # a coef to regulate the range
        self.max_iter = max_iter
        self.is_ci = is_channel_independent
        
        # Keep a scalar loss for TSA (required by this implementation).
        # If you later implement a truly batched CI version, you may switch
        # to reduction="none" inside that specific method only.

    @torch.no_grad()    
    def _zero_order_gradient_estimation(self, x, y_hat, target_idx, delta=1e-3):
        """internal method for calculating pseudo gradient"""
        # searching for the plus side
        x_pos = x.clone()
        x_pos[:, target_idx, :] *= (1 + delta)
        loss_pos = self.loss_fn(self.model(x_pos), y_hat)
        if isinstance(loss_pos, torch.Tensor) and loss_pos.ndim > 0:
            loss_pos = loss_pos.mean()
        
        # searching for the minus side
        x_neg = x.clone()
        x_neg[:, target_idx, :] *= (1 - delta)
        loss_neg = self.loss_fn(self.model(x_neg), y_hat)
        if isinstance(loss_neg, torch.Tensor) and loss_neg.ndim > 0:
            loss_neg = loss_neg.mean()
        
        # calculate gradient and generate perturbation (this method is only for B = 1)
        grad_estimate = (loss_pos - loss_neg) / (2 * delta)
        perturbation = self.epsilon * torch.sign(grad_estimate)
        if not isinstance(perturbation, torch.Tensor):
            perturbation = torch.tensor(perturbation, device=x.device, dtype=x.dtype)

        # Return shape (1, F) so caller can assign into w[:, t, :]
        return perturbation.reshape(1, 1).expand(1, x.shape[2])
    
    def attack(self, x, y_hat):
        self._validate_input(x)
        
        # NOTE:
        # This benchmark framework calls TSA in a strict B=1 loop.
        # Even for channel-independent (CI) models, we use the serial instance
        # attack when batch size is 1. A batched CI implementation can be added
        # later for speed, but is not required for correctness here.
        if self.is_ci and x.shape[0] > 1:
            return self._batched_parallel_attack(x, y_hat)  # optional acceleration

        if x.shape[0] != 1 and not self.is_ci:
            print("[Warning]: the model is identified as CM model. Data points in the same TIME STEP will be addressed in the same way, even samples are different.")
        return self._serial_instance_attack(x, y_hat)
    
    @torch.no_grad()
    def _serial_instance_attack(self, x, y_hat):
        """
        TSA for 1 series
        """
        seq_len = x.shape[1]
        w = torch.zeros_like(x) # the returning matrix (final perturbation)
        S = set() # set which contains candidate points to be attacked
        
        for iteration in range(self.max_iter):
            best_new_idx = -1 # searching from 0 -> seq_len
            max_loss_increase = -float("inf") # set using minimum loss
            
            # find the best time step (Candidate)
            for j in range(seq_len):
                if j in S: continue
                    
                temp_w = w.clone()
                j_perturb = self._zero_order_gradient_estimation(x, y_hat, j)
                temp_w[:, j, :] = j_perturb

                x_adv_test = x * (1 + temp_w)
                current_loss = self.loss_fn(self.model(x_adv_test), y_hat).item()

                if current_loss > max_loss_increase:
                    max_loss_increase = current_loss
                    best_new_idx = j
            
            if best_new_idx != -1:
                S.add(best_new_idx)
                
            # update all perturbation in the set S
            for idx in S:
                w[:, idx, :] = self._zero_order_gradient_estimation(x, y_hat, idx)
            
            # perserve the strongest time step in the attacking
            if len(S) > self.tau:
                losses_for_S = []
                for idx in S:
                    temp_w = torch.zeros_like(w)
                    temp_w[:, idx, :] = w[:, idx, :]
                    x_adv_single = x * (1 + temp_w)
                    single_loss = self.loss_fn(self.model(x_adv_single), y_hat).item()
                    losses_for_S.append((single_loss, idx))
                    
                losses_for_S.sort(reverse=True, key=lambda item: item[0])
                S = set([item[1] for item in losses_for_S[:self.tau]])
            
            # reset candidate time steps not be choiced
            mask = torch.zeros_like(w)
            for idx in S:
                mask[:, idx, :] = 1
            w = w * mask
        
        return w.detach()
    
    def _batched_parallel_attack(self, x, y_hat):
        """Optional batched method for CI models (not implemented)."""
        raise NotImplementedError("Still in coding")
        
# ==============================================
# Observer class for layers in the model
# ==============================================
class TSFMObserver:
    def __init__(self, model, target_layers: list):
        self.model = model
        self.target_layers = target_layers
        self.activations = {}
        self.hooks = []
        
    def _get_hook(self, layer_name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                val = output
            elif hasattr(output, "__getitem__"):
                val = output[0]
            elif hasattr(output, "last_hidden_state"):
                val = output.last_hidden_state
            else:
                return  # 未知の型はエラーを防ぐために無視する

            # 確実に取り出したものがテンソルであることを確認して保存
            if isinstance(val, torch.Tensor):
                self.activations[layer_name] = val.detach().clone()
                
        return hook
    
    def attach(self):
        self.activations.clear()
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                handle = module.register_forward_hook(self._get_hook(name))
                self.hooks.append(handle)
        
    def remove(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.activations.clear()

    def diagnose_divergence(self, x_clean, x_adv):
        self.attach()

        with torch.no_grad():
            self.model(x_clean)
        clean_features = {k: v.clone() for k, v in self.activations.items()}

        with torch.no_grad():
            self.model(x_adv)
        adv_features = {k: v.clone() for k, v in self.activations.items()}

        self.remove()

        divergence_report = {}
        for layer in self.target_layers:
            if layer in clean_features and layer in adv_features:
                clean_tensor = clean_features[layer]
                adv_tensor = adv_features[layer]

                squared_diff = F.mse_loss(clean_tensor, adv_tensor, reduction="none")
                dims_to_reduce_mse = tuple(range(1, squared_diff.ndim))
                mse_per_instance = squared_diff.mean(dim=dims_to_reduce_mse)

                cos_sim = F.cosine_similarity(clean_tensor, adv_tensor, dim=-1)
                mean_cos_sim = cos_sim.mean(dim=1)

                divergence_report[layer] = {
                    "mse": mse_per_instance.cpu().numpy(),
                    "cos_dist": mean_cos_sim.cpu().numpy()
                }

        return divergence_report

if __name__ == "__main__":
    import torch.nn as nn
    
    class MockTSFM(nn.Module):
        def __init__(self):
            super().__init__()
            self.patching_layer = nn.Linear(3, 16)
            self.attn_layer_1 = nn.Linear(16, 16)
            self.attn_layer_2 = nn.Linear(16, 16)
            self.head = nn.Linear(16, 3)

        def forward(self, x):
            x = F.relu(self.patching_layer(x))
            x = F.relu(self.attn_layer_1(x))
            x = F.relu(self.attn_layer_2(x))
            return self.head(x)
        
    target_model = MockTSFM()
    criterion = nn.MSELoss()
    
    # 取一条 B=1 的测试数据 (Length=96, Feature=3)
    batch_x_clean = torch.randn(1, 96, 3) 
    
    # 2. 准备想要监听的模型层名称
    layers_to_watch = ['patching_layer', 'attn_layer_1', 'attn_layer_2']
    observer = TSFMObserver(target_model, layers_to_watch)

    # 3. 实例化攻击器多态武器库
    attackers = [
        GWNAttacker(model=target_model, loss_fn=criterion, scale=0.05),
        TSAttacker(model=target_model, loss_fn=criterion, tau=9, epsilon=0.1, max_iter=5, is_channel_independent=False)
    ]

    # 4. 运行 Benchmark
    print("=== 开始 TSFM 组件级鲁棒性诊断测试 ===")
    with torch.no_grad():
        y_hat_clean = target_model(batch_x_clean)

    for attacker in attackers:
        print(f"\n>> 正在执行攻击: {attacker.__class__.__name__}")
        
        # 多态接口生成对抗扰动
        perturbation = attacker.attack(batch_x_clean, y_hat_clean)
        batch_x_adv = batch_x_clean * (1 + perturbation)
        
        # 呼叫探针生成误差传播报告
        report = observer.diagnose_divergence(batch_x_clean, batch_x_adv)
        
        print("--- 组件级特征偏移量 (MSE) ---")
        prev_mse = 0
        for i, (layer_name, metrics_dict) in enumerate(report.items()):
            # 辞書からそれぞれの配列を取り出す
            mse_array = metrics_dict["mse"]
            cos_dist_array = metrics_dict["cos_dist"]
            
            mean_mse = np.mean(mse_array) # 提取平均偏移量
            mean_cos = np.mean(cos_dist_array) # コサイン距離も計算できる！
            
            # 差异分析：判断该层是"防御层"还是"冗余层"
            delta = mean_mse - prev_mse
            status = ""
            if i > 0:
                if delta < 0:
                    status = "✅ 误差衰减 (可能起到了鲁棒性缓冲作用)"
                elif abs(delta) < 1e-5:
                    status = "⚠️ 误差不变 (可能是可以裁剪的冗余层)"
                else:
                    status = "❌ 误差放大 (模型的脆弱点)"
            
            # 出力に Cosine Distance も追加するとさらに論文っぽくなります
            print(f"[{layer_name}] MSE: {mean_mse:.6f} | CosDist: {mean_cos:.6f} {status}")
            prev_mse = mean_mse