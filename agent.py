#%%

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim

import numpy as np
from copy import deepcopy
from itertools import accumulate
from math import log
import enlighten

from utils import default_args, dkl
from maze import T_Maze, action_size
from buffer import RecurrentReplayBuffer
from models import PVRNN, Actor, Critic



class Agent:
    
    def __init__(self, action_prior = "normal", args = default_args):
        
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
        
        self.pvrnn = PVRNN(self.args)
        self.pvrnn_opt = optim.Adam(self.pvrnn.parameters(), lr=self.args.pvrnn_lr, weight_decay=0)   
        
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
        
        self.memory = RecurrentReplayBuffer(self.args) 
        self.plot_dict = {
            "args" : self.args,
            "title" : "{}_{}".format(self.args.name, self.args.id),
            "rewards" : [], "spot_names" : [], 
            "error" : [], "complexity" : [], 
            "alpha" : [], "actor" : [], 
            "critic_1" : [], "critic_2" : [], 
            "extrinsic" : [], "intrinsic_curiosity" : [], 
            "intrinsic_entropy" : [], 
            "naive" : [], "free" : []}       

    
    
    def training(self):
        manager = enlighten.Manager(width = 150)
        E = manager.counter(total = self.args.episodes, desc = self.plot_dict["title"], unit = "ticks", color = "blue")
        episodes = 0
        while(True):
            self.episode(episodes == 0 or episodes+1 >= self.args.episodes or (episodes) % self.args.keep_data == 0)
            episodes += 1 ; E.update()
            if(episodes >= self.args.episodes): 
                print("\n\nDone training!")
                break
        self.plot_dict["rewards"] = list(accumulate(self.plot_dict["rewards"]))
        
        min_max_dict = {key : [] for key in self.plot_dict.keys()}
        for key in min_max_dict.keys():
            if(not key in ["args", "title", "spot_names"]):
                minimum = None ; maximum = None 
                l = self.plot_dict[key]
                print(key)
                l = deepcopy(l)
                l = [_ for _ in l if _ != None]
                if(l != []):
                    if(  minimum == None):  minimum = min(l)
                    elif(minimum > min(l)): minimum = min(l)
                    if(  maximum == None):  maximum = max(l) 
                    elif(maximum < max(l)): maximum = max(l)
                min_max_dict[key] = (minimum, maximum)
        return(self.plot_dict, min_max_dict)
            
    
    
    
    def episode(self, plot_dict_push, verbose = True):
        done = False ; prev_a = torch.zeros((1, action_size)) ; h = None
        t_maze = T_Maze()
        if(verbose): print("\n\n\n\n\nSTART!\n")
        if(verbose): print(t_maze)
        while(done == False):
            with torch.no_grad():
                o = t_maze.obs()
                a, _, h = self.actor(o, prev_a, h)
                action = a.squeeze(0).tolist()
                r, spot_name, done = t_maze.action(action[0], action[1], verbose)
                next_o = t_maze.obs() ; prev_a = a ; self.steps += 1
                self.memory.push(o, a, r, next_o, done, done)
                
            if((self.steps+1) % self.args.learn_per_steps == 0): 
                plot_data = self.learn(self.args.batch_size)
                if(plot_data != None): 
                    l, e, ie, ic, naive, free = plot_data
                    if(plot_dict_push):
                        self.plot_dict["error"].append(l[0])
                        self.plot_dict["complexity"].append(l[1])
                        self.plot_dict["alpha"].append(l[2])
                        self.plot_dict["actor"].append(l[3])
                        self.plot_dict["critic_1"].append(l[4])
                        self.plot_dict["critic_2"].append(l[5])
                        self.plot_dict["extrinsic"].append(e)
                        self.plot_dict["intrinsic_entropy"].append(ie)
                        self.plot_dict["intrinsic_curiosity"].append(ic)
                        self.plot_dict["naive"].append(naive)
                        self.plot_dict["free"].append(free)
            if(plot_dict_push):
                self.plot_dict["rewards"].append(r)
                self.plot_dict["spot_names"].append(spot_name)        
    
    
    
    def learn(self, batch_size):
        
        batch = self.memory.sample(batch_size)
        if(batch == None): return(None)
        obs, actions, rewards, dones, masks = batch
        next_obs = obs[:,1:] ; obs = obs[:,:-1]
        prev_actions = torch.cat([torch.zeros(actions[:,0].unsqueeze(1).shape), actions[:,:-1]], dim = 1)
        extrinsic_reward = rewards.mean()
        
        print("\n\n")
        print("obs: {}. actions: {}. rewards: {}. dones: {}. masks: {}.".format(obs.shape, actions.shape, rewards.shape, dones.shape, masks.shape))
        print("\n\n")
        
        # Train Forward
        
        
        
        # Train critics
        with torch.no_grad():
            next_actions, log_pis_next, _ = self.actor(next_obs, actions)
            Q_target1_next = self.critic1_target(next_obs, actions, next_actions)
            Q_target2_next = self.critic2_target(next_obs, actions, next_actions)
            Q_target_next = torch.min(Q_target1_next, Q_target2_next)
            if self.args.alpha == None: Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - self.alpha * log_pis_next))
            else:                       Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - self.args.alpha * log_pis_next))
            
        Q_1 = self.critic1(obs, prev_actions, actions)
        critic1_loss = 0.5*F.mse_loss(Q_1*masks, Q_targets*masks)
        self.critic1_opt.zero_grad()
        #critic1_loss.backward()
        #self.critic1_opt.step()
        
        Q_2 = self.critic2(obs, prev_actions, actions)
        critic2_loss = 0.5*F.mse_loss(Q_2*masks, Q_targets*masks)
        self.critic2_opt.zero_grad()
        #critic2_loss.backward()
        #self.critic2_opt.step()
        
        
        
        # Train alpha
        if self.args.alpha == None:
            new_actions, log_pis, _ = self.actor(obs, prev_actions)
            alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy))*masks
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
                new_actions, log_pis, _ = self.actor(obs, prev_actions)

            if self._action_prior == "normal":
                loc = torch.zeros(self.action_size, dtype=torch.float64)
                scale_tril = torch.tensor([[1, 0], [1, 1]], dtype=torch.float64)
                policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
                policy_prior_log_probs = policy_prior.log_prob(new_actions).unsqueeze(-1)
            elif self._action_prior == "uniform":
                policy_prior_log_probs = 0.0
            Q = torch.min(
                self.critic1(obs, prev_actions, new_actions), 
                self.critic2(obs, prev_actions, new_actions)).mean(-1).unsqueeze(-1)
            intrinsic_entropy = torch.mean((alpha * log_pis)*masks).item()
            actor_loss = (alpha * log_pis - policy_prior_log_probs - Q)*masks
            actor_loss = actor_loss.mean() / masks.mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.soft_update(self.critic1, self.critic1_target, self.args.tau)
            self.soft_update(self.critic2, self.critic2_target, self.args.tau)
            
        else:
            intrinsic_entropy = None
            actor_loss = None
            
            
        
        error_loss = 1 ; complexity_loss = 1
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        critic1_loss = log(critic1_loss.item())
        critic2_loss = log(critic2_loss.item())
        losses = [error_loss, complexity_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss]
        
        intrinsic_curiosity = 2
        naive_curiosity = 5
        free_curiosity = 6
        
        return(losses, extrinsic_reward, intrinsic_entropy, intrinsic_curiosity, naive_curiosity, free_curiosity)
        

    
    
    
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def state_dict(self):
        return(
            self.pvrnn.state_dict(),
            self.actor.state_dict(),
            self.critic1.state_dict(),
            self.critic1_target.state_dict(),
            self.critic2.state_dict(),
            self.critic2_target.state_dict())

    def load_state_dict(self, state_dict):
        self.pvrnn.load_state_dict(state_dict[0])
        self.actor.load_state_dict(state_dict[1])
        self.critic1.load_state_dict(state_dict[2])
        self.critic1_target.load_state_dict(state_dict[3])
        self.critic2.load_state_dict(state_dict[4])
        self.critic2_target.load_state_dict(state_dict[5])
        self.memory = RecurrentReplayBuffer(self.args)

    def eval(self):
        self.pvrnn.eval()
        self.actor.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()

    def train(self):
        self.pvrnn.train()
        self.actor.train()
        self.critic1.train()
        self.critic1_target.train()
        self.critic2.train()
        self.critic2_target.train()
        
# %%
