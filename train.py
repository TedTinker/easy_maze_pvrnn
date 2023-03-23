#%%

import torch
import enlighten
from itertools import accumulate
from copy import deepcopy

from utils import default_args
from maze import T_Maze
from agent import Agent



def episode(agent, push = True, verbose = False):
    done = False
    t_maze = T_Maze()
    if(verbose): print("\n\n\n\n\nSTART!\n")
    if(verbose): print(t_maze)
    with torch.no_grad():
        while(done == False):
            o = t_maze.obs()
            a = agent.act(o)
            action = a.squeeze(0).tolist()
            r, spot_name, done = t_maze.action(action[0], action[1], verbose)
            no = t_maze.obs()
            if(push): agent.memory.push(o, a, r, no, done, done)
    return(r, spot_name)



class Trainer():
    def __init__(self, args = default_args, title = None):
        
        self.args = args
        self.title = title
        self.restart()
    
    def restart(self):
        self.e = 0
        self.agent = Agent(args = self.args)
        self.plot_dict = {
            "args" : self.args,
            "title" : self.title,
            "rewards" : [], "spot_names" : [], 
            "error" : [], "complexity" : [], 
            "alpha" : [], "actor" : [], 
            "critic_1" : [], "critic_2" : [], 
            "extrinsic" : [], "intrinsic_curiosity" : [], 
            "intrinsic_entropy" : [], 
            "naive_1" : [], "naive_2" : [], "naive_3" : [], "free" : []}

    def train(self):
        self.agent.train()
        manager = enlighten.Manager(width = 150)
        E = manager.counter(total = self.args.epochs, desc = "{}:".format(self.title), unit = "ticks", color = "blue")
        while(True):
            E.update()
            r, spot_name = episode(self.agent)
            l, e, ic, ie, naive_1, naive_2, naive_3, free = self.agent.learn(batch_size = self.args.batch_size, epochs = self.e)
            self.e += 1
            if(self.e == 1 or self.e >= self.args.epochs or (self.e)%self.args.keep_data==0):
                self.plot_dict["rewards"].append(r)
                self.plot_dict["spot_names"].append(spot_name)
                self.plot_dict["error"].append(l[0][0])
                self.plot_dict["complexity"].append(l[0][1])
                self.plot_dict["alpha"].append(l[0][2])
                self.plot_dict["actor"].append(l[0][3])
                self.plot_dict["critic_1"].append(l[0][4])
                self.plot_dict["critic_2"].append(l[0][5])
                self.plot_dict["extrinsic"].append(e)
                self.plot_dict["intrinsic_curiosity"].append(ic)
                self.plot_dict["intrinsic_entropy"].append(ie)
                self.plot_dict["naive_1"].append(naive_1)
                self.plot_dict["naive_2"].append(naive_2)
                self.plot_dict["naive_3"].append(naive_3)
                self.plot_dict["free"].append(free)
            if(self.e >= self.args.epochs): 
                print("\n\nDone training!")
                break
        self.plot_dict["rewards"] = list(accumulate(self.plot_dict["rewards"]))
        
        min_max_dict = {key : [] for key in self.plot_dict.keys()}
        for key in min_max_dict.keys():
            if(not key in ["args", "title", "spot_names"]):
                minimum = None ; maximum = None 
                l = self.plot_dict[key]
                l = deepcopy(l)
                l = [_ for _ in l if _ != None]
                if(l != []):
                    if(  minimum == None):  minimum = min(l)
                    elif(minimum > min(l)): minimum = min(l)
                    if(  maximum == None):  maximum = max(l) 
                    elif(maximum < max(l)): maximum = max(l)
                min_max_dict[key] = (minimum, maximum)
        return(self.plot_dict, min_max_dict)
    
# %%