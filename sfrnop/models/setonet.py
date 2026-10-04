# ==============================================================================
# Portions of this code are adapted from SetONet by Stepan Tretiakov:
# https://github.com/st-ep/SetONet/tree/main/
#
# Original Copyright (c) 2025 Stepan 
# Licensed under MIT License

# Copyright (c) 2025 Stepan

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================


import torch
from torch import nn, Tensor
from torchinfo import summary
from lightning import LightningModule
import torchmetrics 
import random
import torch.nn.functional as F

def get_batch_mask(batch_size, n_mics, mask_size):
    # Assume img is flattened to (32,1024) --> get random indices
    mics = torch.randperm(mask_size)[:n_mics]

    batch_mask = torch.zeros((mask_size), dtype=torch.uint8)
    batch_mask[mics] = 1
    
    return batch_mask

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between trunk queries and branch sensor encodings.
    Replaces the simple dot-product fusion in DeepONet.
    """
    def __init__(self, 
                 d_model: int, 
                 n_heads: int = 4, 
                 dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, 
                trunk_queries: torch.Tensor,  # (B, N_y, d_model)
                branch_kv: torch.Tensor,      # (B, N_sensors, d_model)
                sensor_mask: torch.Tensor = None  # (B, N_sensors)
               ) -> torch.Tensor:
        """
        trunk_queries: Features at target locations (B, n_points, d_model)
        branch_kv: Encoded sensor information (B, n_sensors, d_model)
        sensor_mask: Optional mask for padded sensors
        """
        # Cross-attention: queries from trunk, keys/values from branch
        attn_out, _ = self.cross_attn(
            query=trunk_queries,
            key=branch_kv,
            value=branch_kv,
            key_padding_mask=~sensor_mask if sensor_mask is not None else None
        )
        
        # Residual + norm
        x = self.norm1(trunk_queries + attn_out)
        
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        
        return x  # (B, n_points, d_model)

class TransformerBranch(nn.Module):
    """
    Full transformer encoder for processing sensor sets.
    Enables richer inter-sensor communication before aggregation.
    """
    def __init__(self, 
                 d_model: int,
                 input_dim: int,
                 output_size: int,
                 n_heads: int = 4, 
                 n_layers: int = 3, 
                 dropout: float = 0.1
                ):
        
        super().__init__()
        
        self.d_model = d_model
        
        # Initial embedding (replaces phi)
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,  # Standard practice
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection (replaces rho)
        self.output_proj = nn.Linear(d_model, output_size)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, N, input_dim) - sensor features
        mask: (B, N) - optional padding mask
        """
        B, N = x.shape[:2]
        
        # Embed inputs
        x = self.input_proj(x)  # (B, N, d_model)
            
        # Self-attention across all sensors + tokens
        out = self.transformer(x)
        
        return self.output_proj(out)  # (B, output_size)
class MLP(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 hidden_features, 
                 hidden_layers,
                 activation=nn.LeakyReLU()):
        super(MLP, self).__init__()

        layers = []

        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(activation)

        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(activation)

        layers.append(nn.Linear(hidden_features, out_features))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class AttentionPool(nn.Module):
    """
    k‑token multi‑head attention aggregator (Set‑Transformer style).
    If `n_tokens = 1` this is identical to the old single‑token pool.
    """
    def __init__(self, d_model: int, n_heads: int = 4, n_tokens: int = 4):
        super().__init__()
        self.n_tokens = n_tokens
        self.query_tokens = nn.Parameter(torch.randn(1, n_tokens, d_model))
        self.layer_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, d_model)   encoded sensors
        returns pooled : (B, n_tokens * d_model) after flattening
        """
        B = x.size(0)
        q = self.query_tokens.expand(B, -1, -1)         # (B, k, d)
        pooled, _ = self.attn(q, x, x)                  # (B, k, d)
        pooled = self.layer_norm(pooled + self.ffn(pooled))              # (B, k, d)
        return pooled.flatten(1)                        # (B, k·d) 
    
class SetONet(LightningModule):

    def __init__(self,
                 input_size_src,      # Dimensionality of sensor location x_i (e.g., 1 for 1D)
                 output_size_src,     # Dimensionality of sensor value u(x_i) (e.g., 1 for scalar)
                 input_size_tgt,      # Dimensionality of trunk input y (e.g., 1 for 1D)
                 output_size_tgt,     # Dimensionality of final output G(u)(y) (e.g., 1 for scalar)
                 p=128,                # Latent dimension for the branch/trunk cross product (Default)
                 phi_hidden_size=256, # Hidden layer size for the phi network (Default)
                 rho_hidden_size=256, # Hidden layer size for the rho network (Default)
                 trunk_hidden_size=256,# Hidden layer size for the trunk network (MLP) (Default)
                 n_trunk_layers=4,    # Number of layers in the trunk network (Default)
                 activation_fn=nn.ReLU, # Activation function
                 use_deeponet_bias=True, # Whether to use a bias term after the cross product
                 phi_output_size=128, # Output dimension of the phi network before aggregation (Default)
                 use_positional_encoding=True, # Flag to enable/disable positional encoding
                 pos_encoding_dim=64, # Dimension for sinusoidal positional encoding
                 pos_encoding_type='sinusoidal', # Type: 'sinusoidal', or 'skip'
                 pos_encoding_max_freq=0.1, # Max frequency/scale for sinusoidal encoding
                 encoding_strategy='concatenate', # Strategy for combining positional and sensor features. Only 'concatenate' is supported.
                 aggregation_type: str = "attention",  # 'mean' or 'attention'
                 loss_fn='nn.MSELoss',
                 n_mics: int | None = None,
                 optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
                 learning_rate: float = 0.001,
    ):
        super().__init__()

        activation_fn = eval(activation_fn)
        self.loss_fn = eval(loss_fn)()

        self.n_mics = n_mics

        # set hyperparameters
        self.input_size_src = input_size_src
        self.output_size_src = output_size_src
        self.input_size_tgt = input_size_tgt
        self.output_size_tgt = output_size_tgt
        # Note: use_positional_encoding is True if type is 'mlp' or 'sinusoidal'
        self.use_positional_encoding = use_positional_encoding and pos_encoding_type != 'skip' # True if type is 'sinusoidal' and use_positional_encoding arg is True
        self.pos_encoding_dim = pos_encoding_dim if self.use_positional_encoding else 0
        self.pos_encoding_type = pos_encoding_type
        self.pos_encoding_max_freq = pos_encoding_max_freq # Store max frequency/scale
        self.encoding_strategy = encoding_strategy

        self.p = p
        self.phi_hidden_size = phi_hidden_size
        self.phi_output_size = phi_output_size
        self.rho_hidden_size = rho_hidden_size
        self.trunk_hidden_size = trunk_hidden_size
        self.n_trunk_layers = n_trunk_layers

        # ---------------------------------------------------------------------
        # Aggregation choice ('mean' | 'sum' | 'attention')
        # ---------------------------------------------------------------------
        self.aggregation = aggregation_type.lower()
        if self.aggregation not in ["mean", "sum", "attention"]:
            raise ValueError("aggregation_type must be one of 'mean', 'sum', or 'attention'")
        self.attention_n_heads = 4 # Default or make configurable if needed

        # Validate encoding strategy
        if self.encoding_strategy != 'concatenate':
            raise ValueError("encoding_strategy must be 'concatenate'. 'film' is no longer supported.")
        # Validate pos_encoding_type
        if self.pos_encoding_type not in ['sinusoidal', 'skip']:
             raise ValueError(f"Unknown pos_encoding_type: {self.pos_encoding_type}. Choose 'sinusoidal' or 'skip'.")
        if self.use_positional_encoding:
            if self.pos_encoding_dim % (2 * self.input_size_src) != 0:
                raise ValueError(
                    f"For sinusoidal encoding, pos_encoding_dim ({self.pos_encoding_dim}) must be divisible by "
                    f"2 * input_size_src ({2 * self.input_size_src})."
                )
        
        if self.use_positional_encoding: # True only for 'sinusoidal' type and if use_positional_encoding arg is True
            phi_input_dim = self.pos_encoding_dim + output_size_src + 1
            # No pos_encoder_mlp is initialized or used for sinusoidal.
        else: # This case handles pos_encoding_type == 'skip' or if use_positional_encoding (arg) was explicitly False
            # Original input: raw position + sensor value
            phi_input_dim = input_size_src + output_size_src + 1
        
        self.branch_network = TransformerBranch(
            d_model=self.phi_output_size,
            input_dim=phi_input_dim,
            output_size=self.output_size_tgt * p,
            n_heads=4,
            n_layers=3
        )


        # --- Trunk Network (MLP) ---
        # Maps y to t_1, ..., t_p (potentially multi-dimensional output_size_tgt)
        trunk_layers = []
        trunk_layers.append(nn.Linear(input_size_tgt, trunk_hidden_size))
        trunk_layers.append(activation_fn())
        for _ in range(n_trunk_layers - 2):
            trunk_layers.append(nn.Linear(trunk_hidden_size, trunk_hidden_size))
            trunk_layers.append(activation_fn())
        trunk_layers.append(nn.Linear(trunk_hidden_size, output_size_tgt * p))

        self.trunk = torch.nn.Sequential(*trunk_layers)
        # --- End Trunk Network ---

        self.cross_fusion = CrossAttentionFusion(
            d_model = self.output_size_tgt * p,
            n_heads=4
        )
        
        self.output_head = nn.Linear(self.output_size_tgt * p, self.output_size_tgt)

        # an optional bias, see equation 2 in the DeepONet paper.
        self.bias = torch.nn.Parameter(torch.randn(output_size_tgt) * 0.1) if use_deeponet_bias else None

        self.metrics = {
            "train": {
                "mse": torchmetrics.MeanSquaredError(),
            },
            "val": {
                "mse": torchmetrics.MeanSquaredError(),
            },
        }

        for split in self.metrics:
            for label, metric in self.metrics[split].items():
                self.register_module(f"metric_{split}_{label}", metric)
        
        self.optimizer_partial = optimizer
        self.learning_rate = learning_rate
    
    def _sinusoidal_encoding(self, coords):
        """Applies fixed sinusoidal encoding to coordinates."""
        # coords shape: (batch_size, n_sensors, input_size_src)
        # Output shape: (batch_size, n_sensors, pos_encoding_dim)

        # Make encoding depend on coords' last dimension (works for both x and y)
        coord_dim = coords.shape[-1]
        # Ensure pos_encoding_dim is divisible by 2*coord_dim
        dims_per_coord = self.pos_encoding_dim // coord_dim
        half_dim = dims_per_coord // 2

        # Frequency bands
        # Shape: (half_dim,)
        div_term = torch.exp(torch.arange(half_dim, device=coords.device) * -(torch.log(torch.tensor(self.pos_encoding_max_freq, device=coords.device)) / half_dim))

        # Expand div_term for broadcasting: (1, 1, 1, half_dim)
        div_term = div_term.reshape(1, 1, 1, half_dim)

        # Expand coords for broadcasting: (batch, n_sensors, input_size_src, 1)
        coords_expanded = coords.unsqueeze(-1)

        # Calculate arguments for sin/cos: (batch, n_sensors, input_size_src, half_dim)
        angles = coords_expanded * div_term

        # Calculate sin and cos embeddings: (batch, n_sensors, input_size_src, half_dim)
        sin_embed = torch.sin(angles)
        cos_embed = torch.cos(angles)

        # Interleave sin and cos and flatten the last two dimensions
        # Shape: (batch, n_sensors, input_size_src, dims_per_coord)
        encoding = torch.cat([sin_embed, cos_embed], dim=-1).reshape(
            coords.shape[0], coords.shape[1], coord_dim, dims_per_coord
        )

        # Reshape to final desired dimension: (batch, n_sensors, pos_encoding_dim)
        encoding = encoding.reshape(coords.shape[0], coords.shape[1], self.pos_encoding_dim)

        return encoding

    def forward_branch(self, xs, us, f, ys=None, sensor_mask=None, sensor_weights=None):
        """
        Forward pass for the Deep Sets Branch.
        Args:
            xs (torch.Tensor): Sensor locations, shape (batch_size, n_sensors, input_size_src)
            us (torch.Tensor): Sensor values, shape (batch_size, n_sensors, output_size_src)
            ys (torch.Tensor | None): Target locations (unused, kept for API compatibility)
            sensor_mask (torch.Tensor | None): (batch_size, n_sensors) bool mask, True = valid
            sensor_weights (torch.Tensor | None): (batch_size, n_sensors) nonnegative weights
        Returns:
            torch.Tensor: Branch output, shape (batch_size, p, output_size_tgt)
        """

        batch_size = xs.shape[0]
        n_sensors = xs.shape[1]

        # Reshape inputs for element-wise processing
        # Shape: (batch * n_sensors, *)
        
        # --- Apply Encoding Strategy ---
        if self.encoding_strategy == 'concatenate':
            if self.use_positional_encoding: # True only for 'sinusoidal' type and if use_positional_encoding arg is True
                encoded_xs_full = self._sinusoidal_encoding(xs)
            else: # Handles 'skip' type or explicitly disabled positional encoding
                encoded_xs_full = xs
        
        branch_input = torch.cat([encoded_xs_full, us, f], dim=-1)
        branch_out = self.branch_network(branch_input, mask=sensor_mask)

        return branch_out

    def forward_trunk(self, ys):
        """
        Forward pass for the Trunk Network.
        Args:
            ys (torch.Tensor): Trunk input locations, shape (batch_size, n_points, input_size_tgt)
        Returns:
            torch.Tensor: Trunk output, shape (batch_size, n_points, p, output_size_tgt)
        """
        if self.use_positional_encoding:
            ys = self._sinusoidal_encoding(ys)
        trunk_out = self.trunk(ys)

        return trunk_out

    def forward(self, xs, us, params, ys, sensor_mask=None, sensor_weights=None):
        """
        Full forward pass for DeepOSet.
        Args:
            xs (torch.Tensor): Sensor locations, shape (batch_size, n_sensors, input_size_src)
            us (torch.Tensor): Sensor values, shape (batch_size, n_sensors, output_size_src)
            ys (torch.Tensor): Trunk input locations, shape (batch_size, n_points, input_size_tgt)
            params (torch.Tensor): Tensor containing frequency, Lx, Ly, Sx, Sy parameters
            sensor_mask (torch.Tensor | None): (batch_size, n_sensors) bool mask, True = valid
            sensor_weights (torch.Tensor | None): (batch_size, n_sensors) nonnegative weights
        Returns:
            torch.Tensor: Predicted output G(u)(y), shape (batch_size, n_points, output_size_tgt)
        """
        # Get branch output: (batch, p, out_tgt)
        b = self.forward_branch(xs, us, f=params[:,0:1,:].repeat(1,xs.shape[1],1), ys=ys, sensor_mask=sensor_mask, sensor_weights=sensor_weights)
        
        # Get trunk output: (batch, n_points, p, out_tgt)
        t = self.forward_trunk(ys)

        G_u_y = self.cross_fusion(t, b)
        G_u_y = self.output_head(G_u_y)

        # optionally add bias
        if self.bias is not None:
            # Bias shape is (output_size_tgt), needs broadcasting to (batch, n_points, output_size_tgt)
            G_u_y = G_u_y + self.bias  # Broadcasting handles the addition

        return G_u_y
    
    def training_step(self, batch: dict[str, Tensor], batch_idx: int):
        sensor_pos = batch['sensor_pos']
        sensor_val = batch['gt_image']

        batch_mask = get_batch_mask(sensor_pos.shape[0], n_mics=random.randint(64,512), mask_size=sensor_pos.shape[1])
        
        masked_indices = torch.nonzero((batch_mask)).reshape(-1,1)[:,0]
        sensor_pos = sensor_pos[:, masked_indices, :]
        sensor_val = sensor_val[:, masked_indices, :]

        branch_freq = batch['cond_params']
        trunk_input_full = batch['trunk_input']
        trunk_target = batch['trunk_target']
        
        # Use as trunk input the points corresponding to the zero entries in the branch input
        zero_indices = torch.nonzero(1-batch_mask).reshape(-1,1)[:,0]
        trunk_input = trunk_input_full[:, zero_indices, :]
        trunk_target = trunk_target[:, zero_indices, :]
        # trunk_input = trunk_input_full

        # Permute to promote invariance to point ordering
        perm = torch.randperm(trunk_input.shape[1])
        trunk_input = trunk_input[:, perm, :]
        trunk_target = trunk_target[:, perm, :]
        
        output = self(sensor_pos, sensor_val, branch_freq, trunk_input)
        self._update_metrics(split="train", output=output, target=trunk_target)
        return self._compute_loss(split="train", output=output, target=trunk_target)
    
    def on_train_epoch_end(self):
        return self._log_metrics(split="train")
    
    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        sensor_pos = batch['sensor_pos']
        sensor_val = batch['gt_image']

        # Generate mask for entire batch
        batch_mask = get_batch_mask(sensor_pos.shape[0], n_mics=random.randint(64,512), mask_size=sensor_pos.shape[1])
            
        masked_indices = torch.nonzero((batch_mask)).reshape(-1,1)[:,0]
        sensor_pos = sensor_pos[:, masked_indices, :]
        sensor_val = sensor_val[:, masked_indices, :]
        branch_freq = batch['cond_params']
        trunk_input = batch['trunk_input']
        trunk_target = batch['trunk_target']

        output = self(sensor_pos, sensor_val, branch_freq, trunk_input)
        self._update_metrics(split="val", output=output, target=trunk_target)
        return self._compute_loss(split="val", output=output, target=trunk_target)
    
    def on_validation_epoch_end(self):
        return self._log_metrics(split="val")

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int):
        sensor_pos = batch['sensor_pos']
        sensor_val = batch['gt_image']

        batch_mask = get_batch_mask(sensor_pos.shape[0], n_mics=self.n_mics, mask_size=sensor_pos.shape[1])
        masked_indices = torch.nonzero((batch_mask)).reshape(-1,1)[:,0]
        sensor_pos = sensor_pos[:, masked_indices, :]
        sensor_val = sensor_val[:, masked_indices, :]
        branch_freq = batch['cond_params']
        trunk_input = batch['trunk_input']
        trunk_target = batch['trunk_target']

        output = self(sensor_pos, sensor_val, branch_freq, trunk_input)
        return {'predictions': output, 'targets': trunk_target}
    
    def _compute_loss(self, split: str, output: Tensor, target: Tensor):
        batch_size = target.shape[0]

        loss = self.loss_fn(output, target)

        self.log(
            f"{split}/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss
    
    def _update_metrics(self, split: str, output: Tensor, target: Tensor):
        split_metrics = self.metrics[split]

        for metric in split_metrics.values():
            metric.update(output, target)
    
    def _log_metrics(self, *, split: str):
        split_metrics = self.metrics[split]

        for metric_name, metric in split_metrics.items():
            self.log(
                f"{split}/{metric_name}",
                metric,
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
            )
            metric.reset()
    
    def configure_optimizers(self):
        optimizer = self.optimizer_partial(self.parameters(), lr=self.learning_rate)
        return optimizer
    
if __name__ == '__main__':
    model = SetONet(
        input_size_src=2,
        output_size_src=1,
        input_size_tgt=2,
        output_size_tgt=1,
        p=32,
        phi_hidden_size=256,
        rho_hidden_size=256,
        trunk_hidden_size=256,
        n_trunk_layers=4,
        activation_fn='nn.LeakyReLU',
        use_deeponet_bias=True,
        phi_output_size=128,
        use_positional_encoding=True,
        pos_encoding_dim=64,
        pos_encoding_type='sinusoidal',
        pos_encoding_max_freq=0.1,
        encoding_strategy='concatenate',
        aggregation_type='attention',
        attention_n_tokens=1,
        branch_head_type='standard',
    )
    
    summary(model)
    
    sensor_pos = torch.rand((32,256,2))
    sensor_val = torch.rand((32,256,1))
    param_input = torch.rand((32,5,1))
    trunk_input = torch.rand((32,1024,2))
    output = model(sensor_pos, sensor_val, param_input, trunk_input)
    print(output.shape)