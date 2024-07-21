import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import collections
import random
from itertools import chain

class ReplayBuffer:
    ''' 经验回放池 '''

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, cost, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, cost, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, cost, next_state, done = zip(*transitions)
        return np.array(state), action, reward, cost, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


# <<< NERUAL NETWORK <<<
#! --- ACTOR NETWORK ---
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

#! --- CRITIC NETWORK ---
class DoubleQCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.Q1_fc1 = nn.Linear(state_dim, hidden_dim)
        self.Q1_fc2 = nn.Linear(hidden_dim, action_dim)

        self.Q2_fc1 = nn.Linear(state_dim, hidden_dim)
        self.Q2_fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        q1 = self.Q1_fc2(F.relu(self.Q1_fc1(x)))
        q2 = self.Q2_fc2(F.relu(self.Q2_fc1(x)))
        return q1, q2

class SafetyCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.QC_fc1 = nn.Linear(state_dim, hidden_dim)
        self.QC_fc2 = nn.Linear(hidden_dim, action_dim)

        self.VC_fc1 = nn.Linear(state_dim, hidden_dim)
        self.VC_fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        qc = self.QC_fc2(F.relu(self.QC_fc1(x)))
        vc = self.VC_fc2(F.relu(self.VC_fc1(x)))
        return qc, vc
    
# >>> NERUAL NETWORK >>>


class WCSACAgent(object):
    """Pytorch-based WCSAC algorithm."""
    def __init__(self,
                 state_dim:int, 
                 hidden_dim:int, 
                 action_dim:int,
                 actor_lr,
                 actor_betas,
                 critic_lr,
                 critic_betas,
                 alpha_lr,
                 alpha_betas,
                 beta_lr,
                 target_entropy,
                 tau,
                 gamma,
                 cost_limit,
                 max_episode_len,
                 risk_level,
                 damp_scale,
                 device):

        self.gamma           = gamma
        self.tau             = tau
        self.cost_limit      = cost_limit
        self.max_episode_len = max_episode_len
        self.device          = device

        # Safety related params
        self.max_episode_len = max_episode_len
        self.cost_limit = cost_limit  # d in Eq. 10
        self.risk_level = risk_level  # alpha in Eq. 9 / risk averse = 0, risk neutral = 1
        normal = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.pdf_cdf = (
            normal.log_prob(normal.icdf(torch.tensor(self.risk_level))).exp() / self.risk_level
        )  # precompute CVaR Value for st. normal distribution
        self.pdf_cdf = self.pdf_cdf.cuda()
        self.damp_scale = damp_scale

        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        
        #! --- REWARD PART ---
        self.critic         = DoubleQCritic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target  = DoubleQCritic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        #! --- COST PART ---
        self.safety_critic = SafetyCritic(state_dim, hidden_dim, action_dim).to(device)
        self.safety_critic_target = SafetyCritic(state_dim, hidden_dim, action_dim).to(device)
        self.safety_critic_target.load_state_dict(self.safety_critic.state_dict())
         
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=actor_betas)
        self.all_critics_optimizer = torch.optim.Adam(
            chain(self.critic.parameters(), self.safety_critic.parameters()),
            lr=critic_lr,
            betas=critic_betas,
        )

        
        # Entropy temperature (beta in the paper)
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=alpha_betas)
        
        # Cost temperature (kappa in the paper)
        self.log_beta = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_beta.requires_grad = True
        self.log_beta_optimizer = torch.optim.Adam([self.log_beta], lr=beta_lr, betas=alpha_betas)
        
        # Set target entropy to -|A|
        self.target_entropy = target_entropy
        
        # Set target cost
        self.target_cost = (self.cost_limit * (1 - self.gamma**self.max_episode_len) / (1 - self.gamma) / self.max_episode_len)
    
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()    

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)
        
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)  # 动作不再是float类型
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        costs = torch.tensor(transition_dict['costs'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        
        # <<< CRITIC NETWORK UPDATE <<<
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)  # 计算熵


        #! --- REWARD PART ---
        current_Q1, current_Q2 = self.critic(states)
        current_Q1 = current_Q1.gather(1, actions)
        current_Q2 = current_Q2.gather(1, actions)       

        target_Q1, target_Q2= self.critic_target(next_states)
        target_V = torch.min(target_Q1, target_Q2)  # q_value
        min_value = torch.sum(next_probs * target_V, dim=1, keepdim=True)
        next_value = min_value + self.log_alpha.exp() * entropy
        target_Q = rewards + self.gamma * next_value * (1 - dones)
        
        #! --- COST PART ---
        current_QC, current_VC = self.safety_critic(states)
        current_QC = current_QC.gather(1, actions)
        current_VC = torch.clamp(current_VC.gather(1, actions), min=1e-8, max=1e8)
        
        # QC, VC TARGETS
        # use next action as an approximation
        next_QC, next_VC = self.safety_critic_target(next_states)
        next_QC = torch.sum(next_probs * next_QC, dim=1, keepdim=True)
        next_VC = torch.clamp(torch.sum(next_probs * next_VC, dim=1, keepdim=True), min=1e-8, max=1e8)
        
        target_QC = costs + self.gamma * next_QC * (1 - dones)
        target_VC = torch.clamp((costs**2) - (current_QC**2) + 2*self.gamma*costs*next_QC + (self.gamma**2)*next_VC + (self.gamma**2)*(next_QC**2), min=1e-8, max=1e8)
        #! --- LOSS FUNCTION ---
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        safety_critic_loss = F.mse_loss(current_QC, target_QC) + torch.mean(current_VC + target_VC - 2*torch.sqrt(current_VC*target_VC))

        total_loss = critic_loss + safety_critic_loss
        
        self.all_critics_optimizer.zero_grad()
        total_loss.backward()
        self.all_critics_optimizer.step()
        
        # >>> CRITIC NETWORK UPDATE >>>
        
        
        # <<< ACTOR NETWORK UPDATE <<<
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        
        #! --- REWARD PART ---
        actor_Q1, actor_Q2 = self.critic(states)
        actor_Q1 = actor_Q1.gather(1, actions)
        actor_Q2 = actor_Q2.gather(1, actions)
        actor_Q = torch.min(actor_Q1, actor_Q2)


        actor_QC, actor_VC = self.safety_critic(states)
        actor_QC = actor_QC.gather(1, actions)
        actor_VC = torch.clamp(actor_VC.gather(1, actions), min=1e-8, max=1e8)

        current_QC, current_VC = self.safety_critic(states)
        current_QC = torch.sum(probs * current_QC, dim=1, keepdim=True)
        current_VC = torch.clamp(torch.sum(probs * current_VC, dim=1, keepdim=True), min=1e-8, max=1e8)

        CVaRa = current_QC + self.pdf_cdf.cuda() * torch.sqrt(current_VC)
        damp = self.damp_scale*torch.mean(self.target_cost - CVaRa)
  
        actor_loss = torch.mean(self.log_alpha.exp() * entropy - actor_Q + (self.log_beta.exp() - damp) * (actor_QC + self.pdf_cdf.cuda() * torch.sqrt(actor_VC)))
        
        self.actor_optimizer.zero_grad() 
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        # >>> ACTOR NETWORK UPDATE >>>

        # <<< TEMPERATURE UPDATE <<<
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        beta_loss = torch.mean((CVaRa - self.target_cost).detach() * self.log_beta.exp())
        self.log_beta_optimizer.zero_grad()
        beta_loss.backward()
        self.log_beta_optimizer.step()
        # >>> ACTOR NETWORK UPDATE >>>
        
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.safety_critic, self.safety_critic_target)