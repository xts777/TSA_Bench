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

class FGSMAttacker(BaseAttacker):
    """
    Fast Gradient Sign Method (FGSM) for Time Series.
    White-box single-step gradient attack.
    """
    def __init__(self, model, loss_fn, epsilon=0.1):
        super().__init__(model, loss_fn)
        self.epsilon = epsilon

    def attack(self, x, y_hat=None):
        self._validate_input(x)
        
        # Add infinitesimal noise to break exact equality (pred == y_hat) when evaluating clean input
        x_init = x.clone().detach()
        x_req = (x_init + 1e-4 * torch.randn_like(x_init)).requires_grad_(True)
        pred = self.model(x_req)
        
        target = y_hat if y_hat is not None else pred.detach()
        loss = self.loss_fn(pred, target)
        
        self.model.zero_grad()
        loss.backward()
        
        grad = x_req.grad.data
        grad_sign = torch.sign(grad)
        
        # Handle exact zero gradient elements
        zero_mask = (grad_sign == 0)
        if zero_mask.any():
            grad_sign[zero_mask] = torch.sign(torch.randn_like(grad[zero_mask]))
            # If sign is still 0 (rare), set to +1
            grad_sign[grad_sign == 0] = 1.0
        
        delta = self.epsilon * grad_sign
        
        # Convert absolute delta to relative perturbation matrix w so x * (1 + w) == x + delta
        w = torch.where(x.abs() > 1e-6, delta / x, torch.zeros_like(x))
        return w.detach()

class PGDAttacker(BaseAttacker):
    """
    Projected Gradient Descent (PGD) for Time Series.
    Iterative white-box attack with L_inf norm constraint.
    """
    def __init__(self, model, loss_fn, epsilon=0.1, alpha=0.02, steps=10, random_start=True):
        super().__init__(model, loss_fn)
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def attack(self, x, y_hat=None):
        self._validate_input(x)
        
        if self.random_start:
            delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        else:
            delta = torch.zeros_like(x)
            
        delta.requires_grad_(True)
        
        for _ in range(self.steps):
            x_adv = x + delta
            pred = self.model(x_adv)
            
            target = y_hat if y_hat is not None else pred.detach()
            loss = self.loss_fn(pred, target)
            
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()
            
            with torch.no_grad():
                grad = delta.grad.data
                delta.data = delta.data + self.alpha * torch.sign(grad)
                # Project back into the L_inf epsilon ball
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            
            delta.requires_grad_(True)
            
        delta_final = delta.detach()
        w = torch.where(x.abs() > 1e-6, delta_final / x, torch.zeros_like(x))
        return w.detach()
    
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
                return  # Ignore unknown output types to prevent errors.

            # Save the value only if it is confirmed to be a tensor.
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
                clean_flat = clean_tensor.flatten(start_dim=1)
                adv_flat = adv_tensor.flatten(start_dim=1)
                diff_flat = adv_flat - clean_flat

                mse_per_instance = diff_flat.pow(2).mean(dim=1)
                cos_sim = F.cosine_similarity(clean_flat, adv_flat, dim=1)

                clean_norm = clean_flat.norm(p=2, dim=1)
                adv_norm = adv_flat.norm(p=2, dim=1)
                norm_ratio = adv_norm / (clean_norm + 1e-8)

                l_inf = diff_flat.abs().max(dim=1).values

                divergence_report[layer] = {
                    "mse": mse_per_instance.cpu().numpy(),
                    "cos_dist": cos_sim.cpu().numpy(),
                    "norm_ratio": norm_ratio.cpu().numpy(),
                    "l_inf": l_inf.cpu().numpy()
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
    
    # Sample one B=1 test example (Length=96, Feature=3).
    batch_x_clean = torch.randn(1, 96, 3) 
    
    # 2. Prepare the model layer names to monitor.
    layers_to_watch = ['patching_layer', 'attn_layer_1', 'attn_layer_2']
    observer = TSFMObserver(target_model, layers_to_watch)

    # 3. Instantiate the attack methods.
    attackers = [
        GWNAttacker(model=target_model, loss_fn=criterion, scale=0.05),
        TSAttacker(model=target_model, loss_fn=criterion, tau=9, epsilon=0.1, max_iter=5, is_channel_independent=False)
    ]

    # 4. Run the benchmark.
    print("=== Starting TSFM component-level robustness diagnostic test ===")
    with torch.no_grad():
        y_hat_clean = target_model(batch_x_clean)

    for attacker in attackers:
        print(f"\n>> Running attack: {attacker.__class__.__name__}")
        
        # Generate adversarial perturbations through the polymorphic interface.
        perturbation = attacker.attack(batch_x_clean, y_hat_clean)
        batch_x_adv = batch_x_clean * (1 + perturbation)
        
        # Call the observer to generate the divergence report.
        report = observer.diagnose_divergence(batch_x_clean, batch_x_adv)
        
        print("--- Component-level feature shift (MSE) ---")
        prev_mse = 0
        for i, (layer_name, metrics_dict) in enumerate(report.items()):
            # Extract the arrays from the dictionary.
            mse_array = metrics_dict["mse"]
            cos_dist_array = metrics_dict["cos_dist"]
            
            mean_mse = np.mean(mse_array)  # Compute the average shift.
            mean_cos = np.mean(cos_dist_array)  # Cosine distance is also available.
            
            # Differential analysis: determine whether the layer is a protective or redundant layer.
            delta = mean_mse - prev_mse
            status = ""
            if i > 0:
                if delta < 0:
                    status = "✅ Error decays (it may act as a robustness buffer)"
                elif abs(delta) < 1e-5:
                    status = "⚠️ Error stays flat (it may be a removable redundant layer)"
                else:
                    status = "❌ Error amplifies (a model weakness)"
            
            # Adding Cosine Distance makes the output look more paper-like.
            print(f"[{layer_name}] MSE: {mean_mse:.6f} | CosDist: {mean_cos:.6f} {status}")
            prev_mse = mean_mse