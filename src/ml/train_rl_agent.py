# ✅ Best Practice: Reinforcement learning fine-tuning loop using policy gradient
# 🧠 ML Signal: This module allows your annotation model to improve beyond supervision using RL exploration

import torch
import torch.optim as optim
import os
import random

from code_analyser.src.ml.model_torch import load_model
from code_analyser.src.ml.dataset_loader import load_local_annotated_dataset
from code_analyser.src.ml.reward_functions import compute_reward
from code_analyser.src.ml.config import MODEL_CONFIG, TRAINING_CONFIG, DATA_PATHS


class RLEnvironment:
    def __init__(self, entries, device):
        self.entries = entries
        self.device = device

    def sample_episode(self, model, max_steps=5):
        model.eval()
        log_probs = []
        rewards = []

        for _ in range(max_steps):
            sample = random.choice(self.entries)
            input_tensor = torch.tensor(sample["input_ids"]).unsqueeze(0).to(self.device)
            label_tensor = torch.tensor(sample["labels"]).float().to(self.device)

            output = model(input_tensor)
            m = torch.distributions.Bernoulli(probs=output)
            action = m.sample()
            log_prob = m.log_prob(action).sum()

            reward = compute_reward(action, label_tensor)
            log_probs.append(log_prob)
            rewards.append(reward)

        return log_probs, rewards


def train_rl_agent(
    episodes=100, steps_per_episode=5, lr=1e-5, gamma=0.99, pretrained_model_path=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    entries = load_local_annotated_dataset(
        tokenizer_name=MODEL_CONFIG["model_name"],
        code_dir=DATA_PATHS["code_dir"],
        annotation_dir=DATA_PATHS["annotation_dir"],
        confidence_threshold=TRAINING_CONFIG["confidence_threshold"],
        max_samples=TRAINING_CONFIG["max_train_samples"],
    )

    model = load_model(
        distilled=False,
        use_hf=True,
        hf_model_name=MODEL_CONFIG["model_name"],
        output_dim=MODEL_CONFIG["output_dim"],
    ).to(device)

    if pretrained_model_path and os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path))
        print(f"✅ Loaded pre-trained model from {pretrained_model_path}")

    env = RLEnvironment(entries, device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for ep in range(episodes):
        log_probs, rewards = env.sample_episode(model, max_steps=steps_per_episode)

        discounted = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted.insert(0, R)

        discounted = torch.tensor(discounted).to(device)
        log_probs = torch.stack(log_probs)
        loss = -torch.sum(log_probs * discounted)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_reward = sum(rewards) / len(rewards)
        print(f"🎮 Episode {ep+1}/{episodes} | Reward: {avg_reward:.3f} | Loss: {loss.item():.4f}")

    save_path = f"{TRAINING_CONFIG['output_dir']}/rl_finetuned_epoch_{episodes}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"🏁 RL fine-tuned model saved to {save_path}")
