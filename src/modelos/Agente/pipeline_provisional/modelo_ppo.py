# modelo_ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor = nn.Linear(hidden_dim, output_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

def train_ppo(env, input_dim, hidden_dim=64, output_dim=3, lr=3e-4, num_epochs=10, eps_clip=0.2, 
              gamma=0.99, num_episodes=10):
    model = ActorCritic(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        states, actions, rewards, log_probs, values = [], [], [], [], []

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            logits, value = model(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done = env.step(action.item())
            if next_state is None:
                break

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value.item())

            state = next_state

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        old_log_probs_tensor = torch.FloatTensor(log_probs)
        values_tensor = torch.FloatTensor(values)

        advantages = returns - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(num_epochs):
            logits, value_pred = model(states_tensor)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_tensor)
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(value_pred.squeeze(), returns)
            entropy = dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[EPISODIO {episode+1}] Pérdida: {loss.item():.4f} - Retorno total: {returns.sum().item():.2f}")

    return model

def test_agent_and_plot(model, env):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, _ = model(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample().item()
        next_state, reward, done = env.step(action)
        total_reward += reward
        if next_state is None:
            break
        state = next_state

    print(f"Recompensa total en la prueba: {total_reward:.2f}")
