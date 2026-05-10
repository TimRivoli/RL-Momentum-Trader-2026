import torch
import torch.nn as nn

class MomentumActorCritic(nn.Module):
    """
    Shared trunk with separate actor and critic heads.

    Design rationale:
    - Small (256 units): prevents overfitting on 31-feature state
    - LayerNorm: stabilizes training when feature scales differ (dispersion vs. pnl %)
    - Separate heads: actor needs to output probabilities, critic needs real-valued V(s)
    - No dropout: noise hurts RL exploration; use entropy bonus instead
    """
    def __init__(self, state_dim: int = 31, action_dim: int = 80):
        super().__init__()
        
        # Shared feature extractor
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Actor head: outputs logits over actions
        self.actor_head = nn.Linear(64, action_dim)
        
        # Critic head: outputs scalar value estimate
        self.critic_head = nn.Linear(64, 1)
    
    def forward(self, state: torch.Tensor):
        features = self.trunk(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        logits, value = self(state)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value
		
