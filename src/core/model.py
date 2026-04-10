import math, torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from huggingface_hub import PyTorchModelHubMixin


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width * scale)
        self.nums   = scale - 1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class ECAPA_TDNN(nn.Module):

    def __init__(self, C):

        super(ECAPA_TDNN, self).__init__()
        self.conv1  = nn.Conv1d(128, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = Bottle2neck(C, C, kernel_size=3, dilation=5, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer5 = nn.Conv1d(4 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 2)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x4 = self.layer4(x + x1 + x2 + x3)

        x = self.layer5(torch.cat((x1, x2, x3, x4), dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t), torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)

        return x


class Wav2Vec2Encoder(nn.Module):
    """SSL encoder based on Hugging Face's Wav2Vec2 model."""

    def __init__(self,
                 model_name_or_path: str = "facebook/wav2vec2-base-960h",
                 output_attentions: bool = False,
                 output_hidden_states: bool = False,
                 normalize_waveform: bool = False):
        """Initialize the Wav2Vec2 encoder.

        Args:
            model_name_or_path: HuggingFace model name or path to local model.
            output_attentions: Whether to output attentions.
            output_hidden_states: Whether to output hidden states.
            normalize_waveform: Whether to normalize the waveform input.
        """
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.normalize_waveform = normalize_waveform

        # Load Wav2Vec2 model
        self.model = Wav2Vec2Model.from_pretrained(
            model_name_or_path,
            gradient_checkpointing=False)
        self.model.config.apply_spec_augment = False
        self.model.masked_spec_embed = None


    def forward(self, x):
        """Forward pass through the Wav2Vec2 encoder.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels)

        Returns:
            Extracted features of shape (batch_size, sequence_length, 1024)
        """
        # Handle shape: convert (batch_size, sequence_length, channels) to (batch_size, sequence_length)
        if x.ndim == 3:
            x = x.squeeze(-1)  # Remove channel dimension if present

        # Normalize input if specified
        if self.normalize_waveform:
            x = x / (torch.max(torch.abs(x), dim=1, keepdim=True)[0] + 1e-8)

        # Wav2Vec2 forward pass
        outputs = self.model(
            x,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
            return_dict=True
        )

        # Extract last hidden state
        last_hidden_state = outputs.last_hidden_state

        return last_hidden_state


class MLPBridge(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = None,
                 dropout: float = 0.1, activation: str = nn.ReLU, n_layers: int = 1):
        """Initialize the MLP bridge.

        Args:
            input_dim: The input dimension from the SSL encoder.
            output_dim: The output dimension for the model.
            hidden_dim: Hidden dimension size. If None, use the average of input and output dims.
            dropout: Dropout probability to apply between layers.
            activation: Activation function to use
            n_layers: Number of MLP layers (repeats of Linear+Activation+Dropout blocks).
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        assert hasattr(activation, 'forward') and callable(getattr(activation, 'forward', None)), "Activation class must have a callable forward() method."
        act_fn = activation

        layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
        # Final output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the bridge.

        Args:
            x: The input tensor from the SSL encoder.

        Returns:
            The transformed tensor.
        """
        return self.mlp(x)


class Spectra0Model(nn.Module, PyTorchModelHubMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self.ssl_encoder = Wav2Vec2Encoder("facebook/wav2vec2-xls-r-300m")
        self.bridge = MLPBridge(1024, 128, hidden_dim=128, activation=nn.SELU())
        self.ecapa_tdnn = ECAPA_TDNN(128)

    def forward(self, x):
        x = self.ssl_encoder(x)
        x = self.bridge(x)
        x = self.ecapa_tdnn(x)
        return x

    @torch.inference_mode()
    def classify(self, x, threshold: float = -1.0625009):
        x = self.forward(x)[:, 1]
        x = (x > threshold).float()
        return x.item()


# Backward-compatible alias used in examples: `from model import spectra_0`
# (class alias, not an instance)
spectra_0 = Spectra0Model
