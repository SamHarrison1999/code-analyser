# 🧠 ML Signal: This model integrates attention mechanisms and transformer embeddings for enhanced code understanding
# ✅ Best Practice: Modularise architecture to support both custom and HuggingFace-based backbones

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from transformers import AutoModel, AutoTokenizer

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


# ✅ Best Practice: Token embedding + self-attention encoder
class AttentionEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # ⚠️ SAST Risk: Ensure masking is applied if padding is used during batching
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ffn(x)
        x = self.norm2(x + ff_output)
        return x


# ✅ Best Practice: Simple actor-critic network split (shared backbone optional)
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared = self.shared(x)
        logits = self.policy_head(shared)
        value = self.value_head(shared)
        return logits, value


class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, entropy_coef=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.model = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits, _ = self.model(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy

    def compute_returns(self, rewards, dones, last_value):
        returns = []
        R = last_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return returns

    def update(self, states, actions, log_probs, values, rewards, dones, entropies):
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        values = torch.stack(values).squeeze().to(self.device)
        entropies = torch.stack(entropies).to(self.device)

        with torch.no_grad():
            next_state = states[-1]
            _, last_value = self.model(next_state.unsqueeze(0))
            returns = self.compute_returns(rewards, dones, last_value.item())
            returns = torch.tensor(returns).to(self.device)

        advantages = returns - values
        value_loss = advantages.pow(2).mean()
        policy_loss = -(log_probs * advantages.detach()).mean()
        entropy_loss = -entropies.mean()

        loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()


class CodeAnnotationModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 3,
        dropout: float = 0.3,
        use_attention: bool = True,
        use_hf: bool = False,
        hf_model_name: str = "microsoft/codebert-base",
    ):
        super().__init__()
        self.use_hf = use_hf and HUGGINGFACE_AVAILABLE

        if self.use_hf:
            # ✅ Best Practice: Allow optional HuggingFace integration
            self.encoder = AutoModel.from_pretrained(hf_model_name)
            self.embed_dim = self.encoder.config.hidden_size
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.attn_encoder = (
                AttentionEncoder(embed_dim) if use_attention else nn.Identity()
            )
            self.embed_dim = embed_dim

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        if self.use_hf:
            # 🧠 ML Signal: HuggingFace models return token-level embeddings; we average them
            x = self.encoder(input_ids=x, attention_mask=(x != 0)).last_hidden_state
            x = torch.mean(x, dim=1)
        else:
            x = self.embedding(x)
            x = self.attn_encoder(x)
            x = torch.mean(x, dim=1)

        x = self.dropout(x)
        return F.softmax(self.classifier(x), dim=1)


# ✅ Best Practice: Provide a lightweight distilled variant for inference
class DistilledCodeAnnotationModel(nn.Module):
    def __init__(self, input_dim: int = 512, output_dim: int = 3):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)


# ✅ Best Practice: Factory function for model initialisation
def load_model(
    distilled: bool = False,
    use_hf: bool = False,
    use_attention: bool = True,
    vocab_size: int = 10000,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    output_dim: int = 3,
    dropout: float = 0.3,
    hf_model_name: str = "microsoft/codebert-base",
) -> nn.Module:
    if distilled:
        return DistilledCodeAnnotationModel(input_dim=embed_dim, output_dim=output_dim)
    return CodeAnnotationModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
        use_attention=use_attention,
        use_hf=use_hf,
        hf_model_name=hf_model_name,
    )
