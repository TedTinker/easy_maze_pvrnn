#%% 

import torch
from torch import nn 
from torch.distributions import Normal
import torch.nn.functional as F
from torchinfo import summary as torch_summary

from utils import default_args, init_weights
from maze import obs_size, action_size



class PVRNN(nn.Module): 
    
    def __init__(self, args = default_args):
        super(PVRNN, self).__init__()
        
        self.args = args
        self.levels = len(args.rnn_speed)
        
        self.hd = torch.nn.ModuleList()
        self.hz = torch.nn.ModuleList()
        self.hhd = torch.nn.ModuleList()
        
        self.zp_mu  = torch.nn.ModuleList()
        self.zp_rho = torch.nn.ModuleList()
        self.zq_mu  = torch.nn.ModuleList()
        self.zq_rho = torch.nn.ModuleList()
        
        for level in range(self.levels):
            self.hd.append(nn.Linear(args.h_size, args.h_size))
            self.hz.append(nn.Linear(args.z_size, args.h_size))
            if(level != 0): self.hhd.append(nn.Linear(args.h_size, args.h_size))
            
            self.zp_mu.append(nn.Sequential(nn.Linear(args.h_size, args.z_size), nn.Tanh()))
            self.zp_rho.append(             nn.Linear(args.h_size, args.z_size))
            self.zq_mu.append(nn.Sequential(nn.Linear(args.h_size + obs_size + action_size, args.z_size), nn.Tanh()))
            self.zq_rho.append(             nn.Linear(args.h_size + obs_size + action_size, args.z_size, args.z_size))
            
        self.o_mu  = nn.Sequential(nn.Linear(args.h_size, obs_size), nn.Tanh())
        self.o_rho = nn.Linear(args.h_size, obs_size)
        
        self.hd.apply(init_weights)
        self.hz.apply(init_weights)
        self.hhd.apply(init_weights)
        self.zp_mu.apply(init_weights)
        self.zp_rho.apply(init_weights)
        self.zq_mu.apply(init_weights)
        self.zq_rho.apply(init_weights)
        self.o_mu.apply(init_weights)
        self.o_rho.apply(init_weights)
        self.to(self.args.device)
        
    def forward(self): pass
        
    def h_d(self, h, d, zq):
        new_h = torch.zeros(h.shape) ; new_d = torch.zeros(d.shape)
        for level in range(self.levels):
            new_h[:,level] += self.hd[level](d[:,level])
            new_h[:,level] += self.hz[level](zq[:,level])
            if(level != 0): new_h[:,:,level] += self.hhd[level](new_d[:,level-1])
            new_h[:,level] = \
                (1 - 1/self.args.rnn_speed[level]) * h[:,:,level] + \
                1/self.args.rnn_speed[level]       * new_h[:,level]
            new_d[:,level] = torch.tanh(new_h[:,level])
        return(new_h, new_d)
    
    def zp(self, d):
        mu  = torch.zeros(d.shape[0], self.levels, self.args.z_size)
        std = torch.zeros(d.shape[0], self.levels, self.args.z_size)
        for level in range(self.levels):
            mu[:,level]  = self.zp_mu[level](d[:,level])
            std[:,level] = torch.log1p(torch.exp(self.zp_rho[level](d[:,level])))
        e = Normal(0, 1).sample(std.shape).to("cuda" if next(self.parameters()).is_cuda else "cpu")
        zp = mu + e * std
        return(zp, mu, std)
    
    def zq(self, d, o, prev_a):
        mu  = torch.zeros(d.shape[0], self.levels, self.args.z_size)
        std = torch.zeros(d.shape[0], self.levels, self.args.z_size)
        for level in range(self.levels):
            x = torch.cat([d[:,level], o, prev_a], dim = -1)
            mu[:,level]  = self.zq_mu[level](x)
            std[:,level] = torch.log1p(torch.exp(self.zq_rho[level](x)))
        e = Normal(0, 1).sample(std.shape).to("cuda" if next(self.parameters()).is_cuda else "cpu")
        zp = mu + e * std
        return(zp, mu, std)
    
    def pred_o(self, d):
        mu  = self.o_mu(d[:,-1])
        std = torch.log1p(torch.exp(self.o_rho(d[:,-1])))
        e = Normal(0, 1).sample(std.shape).to("cuda" if next(self.parameters()).is_cuda else "cpu")
        pred_o = torch.sigmoid(mu + e * std)
        return(pred_o, mu, std, e)
    
    

class Summarizer(nn.Module): 
    
    def __init__(self, args = default_args):
        super(Summarizer, self).__init__()
        
        self.args = args
        self.gru = nn.GRU(
            input_size =  obs_size + action_size,
            hidden_size = args.h_size,
            batch_first = True)
        
        self.gru.apply(init_weights)
        self.to(self.args.device)
        
    def forward(self, obs, prev_a, h):
        x = torch.cat([obs, prev_a], -1)
        h, _ = self.gru(x, h)
        return(h)
    
    
    
class Actor(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args
        
        self.sum = Summarizer(self.args)
        self.lin = nn.Sequential(
            nn.Linear(obs_size, args.h_size),
            nn.LeakyReLU())
        self.mu = nn.Linear(args.h_size, action_size)
        self.rho= nn.Sequential(
            nn.Linear(args.h_size, action_size))

        self.lin.apply(init_weights)
        self.mu.apply(init_weights)
        self.rho.apply(init_weights)
        self.to(self.args.device)

    def forward(self, o, prev_a, h = None, epsilon=1e-6):
        h = self.sum(o, prev_a, h)
        x = self.lin(o)
        mu = self.mu(x)
        std = torch.log1p(torch.exp(self.rho(x)))
        dist = Normal(0, 1)
        e = dist.sample(std.shape).to("cuda" if next(self.parameters()).is_cuda else "cpu")
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        return(action, log_prob, h)
        
        
    
class Critic(nn.Module):

    def __init__(self, args = default_args):
        super(Critic, self).__init__()
        
        self.args = args
        
        self.sum = Summarizer(self.args)
        self.lin = nn.Sequential(
            nn.Linear(args.h_size + action_size, args.h_size),
            nn.LeakyReLU(),
            nn.Linear(args.h_size, 1))

        self.lin.apply(init_weights)
        self.to(args.device)

    def forward(self, o, prev_a, a, h = None):
        h = self.sum(o, prev_a, h)
        x = torch.cat((h, a), dim=-1)
        x = self.lin(x)
        return(x)
    


if __name__ == "__main__":
    
    args = default_args
    args.device = "cpu"
    args.dkl_rate = 1
    
    pvrnn = PVRNN(args)
    
    print("\n\n")
    print(pvrnn)
    print()
    print(torch_summary(pvrnn))
    
    
    
    actor = Actor(args)
    
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, ((3, 1, obs_size), (3, 1, action_size))))
    
    
    
    critic = Critic(args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, ((3, 1, obs_size), (3, 1, action_size), (3, 1, action_size))))
    



# %%
