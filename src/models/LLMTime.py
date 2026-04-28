import torch
import torch.nn as nn
import numpy as np
from utils.llmtime_utils import LLMTimeSerializer, LLMQuantileScaler

class Model(nn.Module):
    """
    LLMTime: Zero-Shot Time-Series Forecasting with LLMs.
    Adapted for the TSA_attack project structure.
    """
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Serialization parameters
        precision = getattr(configs, 'precision', 2)
        self.serializer = LLMTimeSerializer(precision=precision)
        self.scaler = LLMQuantileScaler()
        
        # LLM Settings
        self.model_name = getattr(configs, 'llm_model', 'meta-llama/Llama-2-7b-hf')  # Use Llama as default
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = None
        self.llm = None
        
        # Sampling parameters
        self.num_samples = getattr(configs, 'num_samples', 1)
        self.temp = getattr(configs, 'temperature', 0.7)

    def _init_llm(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            print(f"Loading LLM: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, x):
        """
        x: [Batch, Seq_len, Channel]
        Returns: [Batch, Pred_len, Channel]
        """
        self._init_llm()
        
        batch_size, seq_len, n_channels = x.shape
        x_np = x.detach().cpu().numpy()
        all_preds = np.zeros((batch_size, self.pred_len, n_channels))

        for b in range(batch_size):
            for c in range(n_channels):
                channel_data = x_np[b, :, c]
                
                # 1. Scale data to [0, 1] quantile range
                self.scaler.fit(channel_data)
                scaled_data = self.scaler.transform(channel_data)
                
                # 2. Serialize to string
                input_str = self.serializer.serialize(scaled_data)
                # Ensure it ends with a separator to signal the LLM to continue
                prompt = input_str + self.serializer.separator
                
                # 3. Generate from LLM
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
                
                # We might need to generate multiple samples and average them
                sample_preds = []
                for _ in range(self.num_samples):
                    with torch.no_grad():
                        # We estimate max_new_tokens based on precision and pred_len
                        # Each number is approx (precision + 2) characters including spaces
                        max_tokens = self.pred_len * (self.serializer.precision + 3)
                        
                        output_tokens = self.llm.generate(
                            **inputs, 
                            max_new_tokens=max_tokens,
                            do_sample=(self.temp > 0),
                            temperature=self.temp,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    
                    # Extract only the generated part
                    generated_text = self.tokenizer.decode(output_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    
                    # 4. Deserialize generated string
                    pred_scaled = self.serializer.deserialize(generated_text)
                    
                    # Align to pred_len
                    if len(pred_scaled) > self.pred_len:
                        pred_scaled = pred_scaled[:self.pred_len]
                    elif len(pred_scaled) < self.pred_len:
                        last_val = pred_scaled[-1] if len(pred_scaled) > 0 else scaled_data[-1]
                        pred_scaled = np.pad(pred_scaled, (0, self.pred_len - len(pred_scaled)), constant_values=last_val)
                    
                    # 5. Inverse Scale
                    pred = self.scaler.inverse_transform(pred_scaled)
                    sample_preds.append(pred)
                
                # Average samples for point estimate
                all_preds[b, :, c] = np.mean(sample_preds, axis=0)

        return torch.tensor(all_preds, dtype=torch.float32).to(x.device)
