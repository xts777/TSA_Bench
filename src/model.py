from __future__ import annotations

import os
from argparse import Namespace
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn


class BaseTSFMWrapper(nn.Module):
    """
    Abstract adapter for arbitrary Time-Series Foundation Models (TSFM).

    This wrapper enforces a unified black-box interface:
    - Input:  x with shape (batch, seq_len, features)  == (B, L, F)
    - Output: y_hat with shape (batch, pred_len, features) == (B, L_pred, F)

    It also guarantees:
    - The underlying TSFM is put into eval mode.
    - All parameters of the underlying model have `requires_grad = False`
      so that attacks only optimize over the *input*, not the model weights.
    - Channel-independence metadata is exposed via `is_channel_independent`.
    """

    def __init__(
        self,
        internal_model: nn.Module,
        seq_len: int,
        pred_len: int,
        c_in: int,
        *,
        is_channel_independent: bool,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.c_in = int(c_in)
        self.is_channel_independent: bool = bool(is_channel_independent)

        # Register underlying TSFM and enforce safety constraints
        self.model = internal_model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """
        Unified forward interface.

        Args:
            x: Input time series with shape (B, L, F).

        Returns:
            Tensor: Predicted future series with shape (B, pred_len, F).
        """
        assert x.ndim == 3, f"Expected 3D input (B, L, F), got {x.shape}"
        assert (
            x.shape[1] == self.seq_len
        ), f"seq_len mismatch: expected {self.seq_len}, got {x.shape[1]}"
        assert (
            x.shape[2] == self.c_in
        ), f"feature dim mismatch: expected {self.c_in}, got {x.shape[2]}"

        y = self._forward_internal(x)
        y = self._unwrap_output(y)

        assert y.ndim == 3, f"Wrapped model must return 3D tensor, got {y.shape}"
        assert (
            y.shape[1] == self.pred_len
        ), f"pred_len mismatch: expected {self.pred_len}, got {y.shape[1]}"
        assert (
            y.shape[2] == self.c_in
        ), f"feature dim mismatch in output: expected {self.c_in}, got {y.shape[2]}"
        return y

    def _forward_internal(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Subclasses must implement the actual call into the underlying TSFM.

        Input is always (B, L, F).
        Return value can be:
            - Tensor of shape (B, pred_len, F)
            - Tuple where the first element is the prediction tensor
        """
        raise NotImplementedError

    @staticmethod
    def _unwrap_output(y: Union[Tensor, Tuple[Tensor, ...]]) -> Tensor:
        """
        Strip away auxiliary outputs and keep only the prediction tensor.
        """
        if isinstance(y, tuple):
            if not y:
                raise ValueError("Model returned an empty tuple.")
            y = y[0]
        if not isinstance(y, torch.Tensor):
            raise TypeError(f"Expected Tensor output, got {type(y)}")
        return y


class PatchTSTWrapper(BaseTSFMWrapper):
    """
    Wrapper for the official PatchTST implementation.

    Notes:
    - PatchTST is a channel-independent architecture: `is_channel_independent=True`.
    - We instantiate the official `models/PatchTST.py:Model` and keep it frozen in eval mode.
    - For strict shape unification we route the tensor through the internal backbone that expects
      (B, F, L) and then transpose back to (B, L_pred, F).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        c_in: int,
        *,
        checkpoint_path: Optional[str] = None,
        # Key PatchTST hyperparameters (reasonable defaults)
        n_layers: int = 3,
        n_heads: int = 16,
        d_model: int = 128,
        d_ff: int = 256,
        dropout: float = 0.0,
        fc_dropout: float = 0.0,
        head_dropout: float = 0.0,
        individual: bool = False,
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: Optional[str] = "end",
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        decomposition: bool = False,
        kernel_size: int = 25,
        # Extra official args (kept optional)
        max_seq_len: int = 1024,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: str = "auto",
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type: str = "flatten",
        verbose: bool = False,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Official import path (assumed available in this repo under src/models + src/layers)
        from models.PatchTST import Model as PatchTST_Model

        configs = Namespace(
            enc_in=int(c_in),
            seq_len=int(seq_len),
            pred_len=int(pred_len),
            e_layers=int(n_layers),
            n_heads=int(n_heads),
            d_model=int(d_model),
            d_ff=int(d_ff),
            dropout=float(dropout),
            fc_dropout=float(fc_dropout),
            head_dropout=float(head_dropout),
            individual=bool(individual),
            patch_len=int(patch_len),
            stride=int(stride),
            padding_patch=padding_patch,
            revin=bool(revin),
            affine=bool(affine),
            subtract_last=bool(subtract_last),
            decomposition=bool(decomposition),
            kernel_size=int(kernel_size),
        )
        if extra_config:
            for k, v in extra_config.items():
                setattr(configs, k, v)

        internal = PatchTST_Model(
            configs=configs,
            max_seq_len=int(max_seq_len),
            norm=str(norm),
            attn_dropout=float(attn_dropout),
            act=str(act),
            key_padding_mask=key_padding_mask,
            pre_norm=bool(pre_norm),
            store_attn=bool(store_attn),
            pe=str(pe),
            learn_pe=bool(learn_pe),
            pretrain_head=bool(pretrain_head),
            head_type=head_type,
            verbose=bool(verbose),
        )

        if checkpoint_path:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"checkpoint_path not found: {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            internal.load_state_dict(state, strict=False)

        super().__init__(
            internal_model=internal,
            seq_len=seq_len,
            pred_len=pred_len,
            c_in=c_in,
            is_channel_independent=True,
        )

    def _forward_internal(self, x: Tensor) -> Tensor:
        # PatchTST backbone expects (B, F, L)
        x_t = x.transpose(1, 2)
        if getattr(self.model, "decomposition", False):
            # Use the official forward path when decomposition is enabled.
            # (Its forward accepts (B, L, F) and handles permutes internally.)
            return self.model(x)

        # Default path (decomposition=False): call backbone directly for strict (B,F,L) interface.
        y_t = self.model.model(x_t)  # (B, F, pred_len)
        return y_t.transpose(1, 2)  # (B, pred_len, F)


class iTransformerWrapper(BaseTSFMWrapper):
    """
    Wrapper for the official iTransformer implementation.

    Notes:
    - iTransformer is a channel-mixing architecture: `is_channel_independent=False`.
    - The official model expects inputs shaped (B, L, F) for forecasting.
    - The official forward signature includes time-mark inputs; for plain TS data we pass None.
    - If the official implementation returns a tuple (e.g., with attention), BaseTSFMWrapper
      will keep only the prediction tensor via `_unwrap_output`.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        c_in: int,
        *,
        checkpoint_path: Optional[str] = None,
        # Core iTransformer hyperparameters (defaults aligned with typical TS-Lib configs)
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        factor: int = 1,
        dropout: float = 0.1,
        output_attention: bool = False,
        # Embedding / misc config used by TS-Library implementation
        task_name: str = "long_term_forecast",
        embed: str = "timeF",
        freq: str = "h",
        activation: str = "gelu",
        # Classification-only knobs (kept for completeness)
        num_class: int = 2,
        # Extra config override
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        from models.iTransformer import Model as iTransformer_Model

        configs = Namespace(
            task_name=str(task_name),
            seq_len=int(seq_len),
            pred_len=int(pred_len),
            enc_in=int(c_in),
            dec_in=int(c_in),
            c_out=int(c_in),
            d_model=int(d_model),
            n_heads=int(n_heads),
            e_layers=int(e_layers),
            d_ff=int(d_ff),
            factor=int(factor),
            dropout=float(dropout),
            output_attention=bool(output_attention),
            embed=str(embed),
            freq=str(freq),
            activation=str(activation),
            num_class=int(num_class),
        )
        if extra_config:
            for k, v in extra_config.items():
                setattr(configs, k, v)

        internal = iTransformer_Model(configs)

        if checkpoint_path:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"checkpoint_path not found: {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            internal.load_state_dict(state, strict=False)

        super().__init__(
            internal_model=internal,
            seq_len=seq_len,
            pred_len=pred_len,
            c_in=c_in,
            is_channel_independent=False,
        )

    def _forward_internal(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        # Official iTransformer signature: forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
        # For plain TS forecasting we pass None for mark/decoder inputs.
        return self.model(x, None, None, None)


class MOMENTWrapper(BaseTSFMWrapper):
    """
    Wrapper for MOMENT via the `momentfm` library.

    Unified interface:
        Input : (B, L, F)
        Output: (B, pred_len, F)

    Notes:
    - MOMENTPipeline expects `x_enc` shaped as (B, C, T) where C == n_channels.
      We therefore transpose (B, L, F) -> (B, F, L) before calling the pipeline.
    - The pipeline returns a rich output object (e.g., TimeseriesOutputs) that
      typically contains `.forecast` for forecasting mode. We extract it and
      ensure the returned tensor is (B, pred_len, F).
    - MOMENT is treated as a frozen foundation model (eval + requires_grad=False).
    - MOMENT is considered channel-mixing for attack scheduling:
      `is_channel_independent=False`.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        c_in: int,
        *,
        model_name_or_path: str = "AutonLab/MOMENT-1-large",
        device: Optional[str] = None,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        from momentfm import MOMENTPipeline

        model_kwargs: Dict[str, Any] = {
            "task_name": "forecasting",
            "forecast_horizon": int(pred_len),
        }
        if pipeline_kwargs:
            model_kwargs.update(pipeline_kwargs)

        pipeline = MOMENTPipeline.from_pretrained(
            model_name_or_path,
            model_kwargs=model_kwargs,
        )
        # Required by MOMENT to build task heads etc.
        pipeline.init()

        if device is not None:
            pipeline = pipeline.to(device)

        # Infer the pipeline's expected context length from its patch embedding and head.
        # In official tutorials, MOMENT commonly uses context_length=512 (e.g., patch_len=8, patch_num=64).
        self._moment_context_len = self._infer_moment_context_len(pipeline) or 512

        # BaseTSFMWrapper will freeze parameters of `pipeline` (nn.Module).
        super().__init__(
            internal_model=pipeline,
            seq_len=seq_len,
            pred_len=pred_len,
            c_in=c_in,
            is_channel_independent=False,
        )

    @staticmethod
    def _infer_moment_context_len(pipeline: nn.Module) -> Optional[int]:
        """
        Try to infer MOMENT's expected context length.

        MOMENT patching is fixed-length; the forecasting head typically expects a fixed number
        of patches. If the user-provided `seq_len` differs, we must pad/trim accordingly.
        """
        try:
            patch_len = None
            d_model = None
            head_in = None

            patch_emb = getattr(pipeline, "patch_embedding", None)
            if patch_emb is not None:
                ve = getattr(patch_emb, "value_embedding", None)
                if ve is not None and hasattr(ve, "in_features") and hasattr(ve, "out_features"):
                    patch_len = int(ve.in_features)
                    d_model = int(ve.out_features)

            head = getattr(pipeline, "head", None)
            if head is not None:
                linear = getattr(head, "linear", None)
                if linear is not None and hasattr(linear, "in_features"):
                    head_in = int(linear.in_features)

            if patch_len and d_model and head_in and head_in % d_model == 0:
                patch_num = head_in // d_model
                return int(patch_num * patch_len)
        except Exception:
            return None
        return None

    def _forward_internal(self, x: Tensor) -> Tensor:
        # MOMENT expects (B, C, T)
        x_enc_in = x.transpose(1, 2)  # (B, F, L)

        # Pad/trim to MOMENT's expected context length.
        ctx = int(self._moment_context_len)
        b, c, l = x_enc_in.shape
        if l == ctx:
            x_enc = x_enc_in
            input_mask = None
        elif l > ctx:
            # Keep the most recent context.
            x_enc = x_enc_in[:, :, -ctx:]
            input_mask = None
        else:
            # Left-pad so the most recent steps align at the end.
            pad = ctx - l
            x_enc = torch.zeros((b, c, ctx), device=x_enc_in.device, dtype=x_enc_in.dtype)
            x_enc[:, :, -l:] = x_enc_in
            input_mask = torch.zeros((b, ctx), device=x_enc_in.device, dtype=x_enc_in.dtype)
            input_mask[:, -l:] = 1.0

        if input_mask is None:
            out = self.model(x_enc=x_enc)
        else:
            out = self.model(x_enc=x_enc, input_mask=input_mask)

        # Extract forecast tensor
        forecast = None
        if isinstance(out, dict):
            forecast = out.get("forecast", None)
        else:
            forecast = getattr(out, "forecast", None)

        if forecast is None:
            raise ValueError(f"MOMENT output does not contain `forecast`. Got type={type(out)}")
        if not isinstance(forecast, torch.Tensor):
            raise TypeError(f"MOMENT forecast is not a Tensor: {type(forecast)}")

        # Expected forecast: (B, pred_len, C). Some versions may return (B, C, pred_len).
        if forecast.ndim != 3:
            raise ValueError(f"Unexpected forecast ndim: {forecast.ndim}, shape={tuple(forecast.shape)}")
        if int(forecast.shape[1]) == self.c_in and int(forecast.shape[2]) == self.pred_len:
            forecast = forecast.transpose(1, 2)
        return forecast


def load_tsfm_wrapper(model_name: str, seq_len: int, pred_len: int, c_in: int, checkpoint_path: Optional[str] = None) -> BaseTSFMWrapper:
    """
    Factory function used by main.py to obtain a wrapped TSFM.

    Args:
        model_name: Name of the TSFM family (e.g., "PatchTST", "iTransformer").
        seq_len:   Input sequence length L.
        pred_len:  Prediction horizon L_pred.
        c_in:      Number of input channels / features F.

    Returns:
        An instance of `BaseTSFMWrapper` (actually a concrete subclass).

    Raises:
        ValueError: If the given model name is not supported.
    """
    name = model_name.lower()
    if name in {"patchtst", "patch-tst", "patch_tst"}:
        return PatchTSTWrapper(seq_len=seq_len, pred_len=pred_len, c_in=c_in, checkpoint_path=checkpoint_path)
    if name in {"itransformer", "i-transformer", "i_transformer"}:
        return iTransformerWrapper(seq_len=seq_len, pred_len=pred_len, c_in=c_in, checkpoint_path=checkpoint_path)
    if name.startswith("moment"):
        model_id = "AutonLab/MOMENT-1-large"
        if ":" in model_name:
            _, rest = model_name.split(":", 1)
            rest = rest.strip()
            if rest:
                model_id = rest
        return MOMENTWrapper(seq_len=seq_len, pred_len=pred_len, c_in=c_in, model_name_or_path=model_id)

    raise ValueError(f"Unsupported TSFM model name: {model_name}")