#%%

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim

import numpy as np
from math import log

from utils import default_args, dkl
from buffer import RecurrentReplayBuffer
from models import Forward, Actor, Critic



class Agent:
    
    def __init__(self, action_prior="normal", args = default_args):
        
        self.args = args
        self.steps = 0
        self.action_size = 2
        
        self.target_entropy = self.args.target_entropy # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_opt = optim.Adam(params=[self.log_alpha], lr=self.args.alpha_lr, weight_decay=0) 
        self._action_prior = action_prior
        
        self.eta = 1
        self.log_eta = torch.tensor([0.0], requires_grad=True)
        
        self.forward = Forward(self.args)
        self.forward_opt = optim.Adam(self.forward.parameters(), lr=self.args.forward_lr, weight_decay=0)   
        
        self.clone = Forward(self.args)
        clone_params = self.forward.state_dict() # Use just parameters with mu, rho
        self.clone_opt = optim.Adam(self.clone.parameters(), lr=self.args.forward_lr, weight_decay=0)   
                           
        self.actor = Actor(self.args)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr, weight_decay=0)     
        
        self.critic1 = Critic(self.args)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=self.args.critic_lr, weight_decay=0)
        self.critic1_target = Critic(self.args)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(self.args)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=self.args.critic_lr, weight_decay=0) 
        self.critic2_target = Critic(self.args)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.restart_memory()
        
    def restart_memory(self):
        self.memory = RecurrentReplayBuffer(self.args)

    def act(self, obs):
        action, _ = self.actor(obs)
        return(action[0])
    
    
    
    def learn(self, batch_size, epochs):
                
        self.steps += 1

        obs, actions, rewards, dones, masks = self.memory.sample(batch_size)
        
        #print("\n\n")
        #print(obs.shape, actions.shape, rewards.shape, dones.shape, masks.shape)
        #print("\n\n")
        
        next_obs = obs[:,1:]
        obs = obs[:,:-1]
        
        print(actions[:,0].unsqueeze(1).shape, actions[:,:-1].shape)
        prev_actions = torch.cat([torch.zeros(actions[:,0].unsqueeze(1).shape), actions[:,:-1]], dim = 1)
        
        
                            
        # Train forward and clone
        def train_forward(forward, opt):
            pred_obs, mu_b, std_b = forward(obs, actions)   
            errors = F.mse_loss(pred_obs, next_obs.detach(), reduction = "none").sum(-1).unsqueeze(-1)
            complexity = dkl(mu_b, std_b, torch.zeros(mu_b.shape),  self.args.sigma * torch.ones(std_b.shape))
                    
            errors = errors * masks.detach()
            error_loss = errors.mean()
            complexity = complexity * masks.detach()
            complexity_loss = complexity.mean()
            forward_loss = error_loss + self.args.beta * complexity_loss
            if(self.args.beta == 0): complexity = None ; complexity_loss = None
            
            opt.zero_grad()
            forward_loss.backward()
            opt.step()
            return(mu_b, std_b, errors, error_loss, complexity_loss)
        
        self.clone.load_state_dict(self.forward.state_dict())
        mu_b, std_b, errors, error_loss, complexity_loss = \
            train_forward(self.forward, self.forward_opt)
        train_forward(self.clone, self.clone_opt)
                        
        
        
        # Get curiosity  
        naive_1_curiosity = self.args.naive_1_eta * errors
        
        _, mu_a, std_a = self.forward(obs, actions)    
        naive_2_curiosity = self.args.naive_2_eta * torch.abs(mu_a - mu_b).mean(-1).unsqueeze(-1)
        naive_3_curiosity = self.args.naive_2_eta * torch.pow(mu_a - mu_b, 2).mean(-1).unsqueeze(-1)
        
        dkl_changes = dkl(mu_a, std_a, mu_b, std_b).mean(-1).unsqueeze(-1)
        free_curiosity = self.args.free_eta * dkl_changes   
        
        if(self.args.curiosity == "naive_1"):   curiosity = naive_1_curiosity
        elif(self.args.curiosity == "naive_2"): curiosity = naive_2_curiosity
        elif(self.args.curiosity == "naive_3"): curiosity = naive_3_curiosity
        elif(self.args.curiosity == "free"):    curiosity = free_curiosity
        else:                                   curiosity = torch.zeros(rewards.shape)
        curiosity *= masks.detach() 
        
        extrinsic = torch.mean(rewards).item()
        intrinsic_curiosity = curiosity.mean().item()
        if(intrinsic_curiosity != 0): intrinsic_curiosity = log(intrinsic_curiosity) 
        rewards += curiosity
        
        
                
        # Train critics
        next_action, log_pis_next = self.actor(next_obs.detach())
        print(actions.shape, next_action.shape)
        Q_target1_next = self.critic1_target(next_obs.detach(), actions.detach(), next_action.detach())
        Q_target2_next = self.critic2_target(next_obs.detach(), actions.detach(), next_action.detach())
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)
        if self.args.alpha == None: Q_targets = rewards.cpu() + (self.args.GAMMA * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.cpu()))
        else:                       Q_targets = rewards.cpu() + (self.args.GAMMA * (1 - dones.cpu()) * (Q_target_next.cpu() - self.args.alpha * log_pis_next.cpu()))
        
        Q_1 = self.critic1(obs.detach(), prev_actions, actions.detach()).cpu()
        critic1_loss = 0.5*F.mse_loss(Q_1*masks.detach().cpu(), Q_targets.detach()*masks.detach().cpu())
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        
        Q_2 = self.critic2(obs.detach(), prev_actions, actions.detach()).cpu()
        critic2_loss = 0.5*F.mse_loss(Q_2*masks.detach().cpu(), Q_targets.detach()*masks.detach().cpu())
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()
        
        
        
        # Train alpha
        if self.args.alpha == None:
            actions_pred, log_pis = self.actor(obs.detach())
            alpha_loss = -(self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu())*masks.detach().cpu()
            alpha_loss = alpha_loss.mean() / masks.mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = torch.exp(self.log_alpha) 
        else:
            alpha_loss = None
            
            
        
        # Train actor
        if self.steps % self.args.d == 0:
            if self.args.alpha == None: alpha = self.alpha 
            else:                       
                alpha = self.args.alpha
                actions_pred, log_pis = self.actor(obs.detach())

            if self._action_prior == "normal":
                loc = torch.zeros(self.action_size, dtype=torch.float64)
                scale_tril = torch.tensor([[1, 0], [1, 1]], dtype=torch.float64)
                policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
                policy_prior_log_probs = policy_prior.log_prob(actions_pred.cpu()).unsqueeze(-1)
            elif self._action_prior == "uniform":
                policy_prior_log_probs = 0.0
            Q = torch.min(
                self.critic1(obs.detach(), prev_actions, actions_pred), 
                self.critic2(obs.detach(), prev_actions, actions_pred)).mean(-1).unsqueeze(-1)
            intrinsic_entropy = torch.mean((alpha * log_pis.cpu())*masks.detach().cpu()).item()
            actor_loss = (alpha * log_pis.cpu() - policy_prior_log_probs - Q.cpu())*masks.detach().cpu()
            actor_loss = actor_loss.mean() / masks.mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.soft_update(self.critic1, self.critic1_target, self.args.tau)
            self.soft_update(self.critic2, self.critic2_target, self.args.tau)
            
        else:
            intrinsic_entropy = None
            actor_loss = None
        
        if(error_loss != None): 
            error_loss = error_loss.item()
        if(complexity_loss != None): 
            complexity_loss = complexity_loss.item()
            if(self.args.beta == 0): complexity_loss = None
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        if(critic1_loss != None): 
            critic1_loss = critic1_loss.item()
            critic1_loss = log(critic1_loss) if critic1_loss > 0 else critic1_loss
        if(critic2_loss != None): 
            critic2_loss = critic2_loss.item()
            critic2_loss = log(critic2_loss) if critic2_loss > 0 else critic2_loss
        losses = np.array([[error_loss, complexity_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss]])
        
        naive_1_curiosity = naive_1_curiosity.mean().item()
        naive_2_curiosity = naive_2_curiosity.mean().item()
        naive_3_curiosity = naive_3_curiosity.mean().item()
        free_curiosity = free_curiosity.mean().item()
        if(free_curiosity == 0): free_curiosity = None
        return(losses, extrinsic, intrinsic_curiosity, intrinsic_entropy, naive_1_curiosity, naive_2_curiosity, naive_3_curiosity, free_curiosity)
                     
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def state_dict(self):
        return(
            self.forward.state_dict(),
            self.actor.state_dict(),
            self.critic1.state_dict(),
            self.critic1_target.state_dict(),
            self.critic2.state_dict(),
            self.critic2_target.state_dict())

    def load_state_dict(self, state_dict):
        self.forward.load_state_dict(state_dict[0])
        self.actor.load_state_dict(state_dict[1])
        self.critic1.load_state_dict(state_dict[2])
        self.critic1_target.load_state_dict(state_dict[3])
        self.critic2.load_state_dict(state_dict[4])
        self.critic2_target.load_state_dict(state_dict[5])
        self.memory = RecurrentReplayBuffer(self.args)

    def eval(self):
        self.forward.eval()
        self.actor.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()

    def train(self):
        self.forward.train()
        self.actor.train()
        self.critic1.train()
        self.critic1_target.train()
        self.critic2.train()
        self.critic2_target.train()
        
# %%
