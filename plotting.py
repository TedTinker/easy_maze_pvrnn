import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal

import matplotlib.pyplot as plt 
import numpy as np
import datetime 
from math import log

from utils import duration



def get_quantiles(plot_dict, name):
    xs = [i for i, x in enumerate(plot_dict[name][0]) if x != None]
    print(name)
    for j in plot_dict[name]: print(len(j))
    lists = np.array(plot_dict[name], dtype=float)    
    lists = lists[:,xs]
    quantile_dict = {"xs" : [x * plot_dict["args"][0].keep_data for x in xs]}
    quantile_dict["min"] = np.min(lists, 0)
    quantile_dict["q10"] = np.quantile(lists, .1, 0)
    quantile_dict["q20"] = np.quantile(lists, .2, 0)
    quantile_dict["q30"] = np.quantile(lists, .3, 0)
    quantile_dict["q40"] = np.quantile(lists, .4, 0)
    quantile_dict["med"] = np.quantile(lists, .5, 0)
    quantile_dict["q60"] = np.quantile(lists, .6, 0)
    quantile_dict["q70"] = np.quantile(lists, .7, 0)
    quantile_dict["q80"] = np.quantile(lists, .8, 0)
    quantile_dict["q90"] = np.quantile(lists, .9, 0)
    quantile_dict["max"] = np.max(lists, 0)
    return(quantile_dict)

def get_logs(quantile_dict):
    for key in quantile_dict.keys():
        if(key != "xs"): quantile_dict[key] = np.log(quantile_dict[key])
    return(quantile_dict)

def log(x):
    try: x = log(x)
    except: pass
    return(x)


def awesome_plot(here, quantile_dict, color, label, min_max = None, line_transparency = .9, fill_transparency = .1):
    here.fill_between(quantile_dict["xs"], quantile_dict["min"], quantile_dict["max"], color = color, alpha = fill_transparency, linewidth = 0)
    here.fill_between(quantile_dict["xs"], quantile_dict["q10"], quantile_dict["q90"], color = color, alpha = fill_transparency, linewidth = 0)    
    here.fill_between(quantile_dict["xs"], quantile_dict["q20"], quantile_dict["q80"], color = color, alpha = fill_transparency, linewidth = 0)
    here.fill_between(quantile_dict["xs"], quantile_dict["q30"], quantile_dict["q70"], color = color, alpha = fill_transparency, linewidth = 0)
    here.fill_between(quantile_dict["xs"], quantile_dict["q40"], quantile_dict["q60"], color = color, alpha = fill_transparency, linewidth = 0)
    handle, = here.plot(quantile_dict["xs"], quantile_dict["med"], color = color, alpha = line_transparency, label = label)
    if(min_max != None and min_max[0] != min_max[1]): here.set_ylim([min_max[0], min_max[1]])
    return(handle)
    
    
    
def many_min_max(min_max_list):
    mins = [min_max[0] for min_max in min_max_list if min_max[0] != None]
    maxs = [min_max[1] for min_max in min_max_list if min_max[1] != None]
    return((min(mins), max(maxs)))



def plots(plot_dicts, min_max_dict):
    fig, axs = plt.subplots(15, len(plot_dicts), figsize = (10*len(plot_dicts), 100))
                
    for i, plot_dict in enumerate(plot_dicts):
    
        # Cumulative rewards
        rew_dict = get_quantiles(plot_dict, "rewards")
        max_rewards = [10*x/plot_dict["args"][0].keep_data for x in range(rew_dict["xs"][-1])]
        min_rewards = [-1*x/plot_dict["args"][0].keep_data for x in range(rew_dict["xs"][-1])]
        
        ax = axs[0,i] if len(plot_dicts) > 1 else axs[0]
        awesome_plot(ax, rew_dict, "turquoise", "Reward")
        ax.axhline(y = 0, color = 'black', linestyle = '--', alpha = .2)
        ax.set_ylabel("Reward")
        ax.set_title(plot_dict["title"] + "\nCumulative Rewards")
        
        ax = axs[1,i] if len(plot_dicts) > 1 else axs[1]
        awesome_plot(ax, rew_dict, "turquoise", "Reward", min_max_dict["rewards"])
        ax.axhline(y = 0, color = "black", linestyle = '--', alpha = .2)
        ax.plot(max_rewards, color = "black", label = "Max Reward")
        ax.plot(min_rewards, color = "black", label = "Max Reward")
        ax.set_ylabel("Reward")
        ax.set_title(plot_dict["title"] + "\nCumulative Rewards, shared min/max")
    
    
    
        # Ending spot
        ax = axs[2,i] if len(plot_dicts) > 1 else axs[2]
        spot_names = np.array([spot_names for spot_names in plot_dict["spot_names"]])
        agents = spot_names.shape[0] ; xs = list(range(spot_names.shape[1]))        
        kinds = ["NONE", "BAD", "GOOD"]
        
        for j, kind in enumerate(kinds):
            counts = np.count_nonzero(spot_names == kind, 0)
            counts = [count + (j*agents*1.1) for count in counts]
            ax.fill_between(xs, [j*agents*1.1 for _ in xs], counts, color = "black", linewidth = 0)
            if(j != len(kinds)-1):
                ax.plot(xs, [agents*1.05 + j*agents*1.1 for _ in xs], color = "black", linestyle = "--")
        ax.set_yticks([(2*j+1)*agents*1.1/2 for j in range(len(kinds))], kinds, rotation='vertical')
        ax.tick_params(left = False)
        ax.set_ylim([-1, len(kinds)*agents*1.1])
        ax.set_ylabel("Ending Spot")
        ax.set_title(plot_dict["title"] + "\nEnding Spots")
        
        
        
        # Forward Losses
        error_dict = get_quantiles(plot_dict, "error")
        comp_dict  = get_quantiles(plot_dict, "complexity")
        min_max = many_min_max([min_max_dict["error"], min_max_dict["complexity"]])
        
        ax = axs[3,i] if len(plot_dicts) > 1 else axs[3]
        h1 = awesome_plot(ax, error_dict, "green", "Error")
        ax.set_ylabel("Loss")
        h2 = awesome_plot(ax, comp_dict, "red", "Complexity")
        ax.legend(handles = [h1, h2])
        ax.set_title(plot_dict["title"] + "\nForward Losses")
        
        ax = axs[4,i] if len(plot_dicts) > 1 else axs[4]
        h1 = awesome_plot(ax, error_dict, "green", "Error", min_max)
        ax.set_ylabel("Loss")
        h2 = awesome_plot(ax, comp_dict, "red", "Complexity", min_max)
        ax.legend(handles = [h1, h2])
        ax.set_title(plot_dict["title"] + "\nForward Losses, shared min/max")
        
        
        
        # Log Forward Losses
        log_error_dict = get_logs(error_dict)
        log_comp_dict  = get_logs(comp_dict)
        min_max = (log(min_max[0]), log(min_max[1]))
        
        ax = axs[5,i] if len(plot_dicts) > 1 else axs[5]
        h1 = awesome_plot(ax, log_error_dict, "green", "log Error")
        ax.set_ylabel("log Loss")
        h2 = awesome_plot(ax, log_comp_dict, "red", "log Complexity")
        ax.legend(handles = [h1, h2])
        ax.set_title(plot_dict["title"] + "\nlog Forward Losses")
        
        ax = axs[6,i] if len(plot_dicts) > 1 else axs[6]
        h1 = awesome_plot(ax, log_error_dict, "green", "log Error", min_max)
        ax.set_ylabel("log Loss")
        h2 = awesome_plot(ax, log_comp_dict, "red", "log Complexity", min_max)
        ax.legend(handles = [h1, h2])
        ax.set_title(plot_dict["title"] + "\nlog Forward Losses, shared min/max")
        
        
        
        # Other Losses
        alpha_dict = get_quantiles(plot_dict, "alpha")
        actor_dict = get_quantiles(plot_dict, "actor")
        crit1_dict = get_quantiles(plot_dict, "critic_1")
        crit2_dict = get_quantiles(plot_dict, "critic_2")
        
        ax = axs[7,i] if len(plot_dicts) > 1 else axs[7]
        h1 = awesome_plot(ax, actor_dict, "red", "Actor")
        ax.set_ylabel("Actor Loss")
        ax2 = ax.twinx()
        h2 = awesome_plot(ax2, crit1_dict, "blue", "log Critic")
        awesome_plot(ax2, crit2_dict, "blue", "log Critic")
        ax2.set_ylabel("log Critic Losses")
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.08))
        h3 = awesome_plot(ax3, alpha_dict, "black", "Alpha")
        ax3.set_ylabel("Alpha Loss")
        ax.legend(handles = [h1, h2, h3])
        ax.set_title(plot_dict["title"] + "\nOther Losses")
        
        ax = axs[8,i] if len(plot_dicts) > 1 else axs[8]
        min_max = many_min_max([min_max_dict["critic_1"], min_max_dict["critic_2"]])
        h1 = awesome_plot(ax, actor_dict, "red", "Actor", min_max_dict["actor"])
        ax.set_ylabel("Actor Loss")
        ax2 = ax.twinx()
        h2 = awesome_plot(ax2, crit1_dict, "blue", "log Critic", min_max)
        awesome_plot(ax2, crit2_dict, "blue", "log Critic", min_max)
        ax2.set_ylabel("log Critic Losses")
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.08))
        h3 = awesome_plot(ax3, alpha_dict, "black", "Alpha", min_max_dict["alpha"])
        ax3.set_ylabel("Alpha Loss")
        ax.legend(handles = [h1, h2, h3])
        ax.set_title(plot_dict["title"] + "\nOther Losses, shared min/max")
        
        
        
        # Extrinsic and Intrinsic rewards
        ext_dict = get_quantiles(plot_dict, "extrinsic")
        cur_dict = get_quantiles(plot_dict, "intrinsic_curiosity")
        ent_dict = get_quantiles(plot_dict, "intrinsic_entropy")
        
        ax = axs[9,i] if len(plot_dicts) > 1 else axs[9]
        handles = []
        handles.append(awesome_plot(ax, ext_dict, "red", "Extrinsic"))
        ax.set_ylabel("Extrinsic")
        if((cur_dict["min"] != cur_dict["max"]).all()):
            ax2 = ax.twinx()
            handles.append(awesome_plot(ax2, cur_dict, "green", "Curiosity"))
            ax2.set_ylabel("Curiosity")
        if((ent_dict["min"] != ent_dict["max"]).all()):
            ax3 = ax.twinx()
            ax3.spines["right"].set_position(("axes", 1.08))
            handles.append(awesome_plot(ax3, ent_dict, "black", "Entropy"))
            ax3.set_ylabel("Entropy")
        ax.legend(handles = handles)
        ax.set_title(plot_dict["title"] + "\nExtrinsic and Intrinsic Rewards")
        
        ax = axs[10,i] if len(plot_dicts) > 1 else axs[10]
        handles = []
        handles.append(awesome_plot(ax, ext_dict, "red", "Extrinsic", min_max_dict["extrinsic"]))
        ax.set_ylabel("Extrinsic")
        if((cur_dict["min"] != cur_dict["max"]).all()):
            ax2 = ax.twinx()
            handles.append(awesome_plot(ax2, cur_dict, "green", "Curiosity", min_max_dict["intrinsic_curiosity"]))
            ax2.set_ylabel("Curiosity")
        if((ent_dict["min"] != ent_dict["max"]).all()):
            ax3 = ax.twinx()
            ax3.spines["right"].set_position(("axes", 1.08))
            handles.append(awesome_plot(ax3, ent_dict, "black", "Entropy", min_max_dict["intrinsic_entropy"]))
            ax3.set_ylabel("Entropy")
        ax.legend(handles = handles)
        ax.set_title(plot_dict["title"] + "\nExtrinsic and Intrinsic Rewards, shared min/max")
        
        
        
        # Curiosities
        naive_dict = get_quantiles(plot_dict, "naive")
        free_dict = get_quantiles(plot_dict, "free")
        min_max = many_min_max([min_max_dict["naive"], min_max_dict["free"]])
        
        ax = axs[11,i] if len(plot_dicts) > 1 else axs[11]
        awesome_plot(ax, naive_dict, "green", "Naive")
        awesome_plot(ax, free_dict, "red", "Free")
        ax.set_ylabel("Curiosity")
        ax.legend()
        ax.set_title(plot_dict["title"] + "\nPossible Curiosities")
        
        ax = axs[12,i] if len(plot_dicts) > 1 else axs[12]
        awesome_plot(ax, naive_dict, "green", "Naive", min_max)
        awesome_plot(ax, free_dict, "red", "Free", min_max)
        ax.set_ylabel("Curiosity")
        ax.legend()
        ax.set_title(plot_dict["title"] + "\nPossible Curiosities, shared min/max")
        
        
        
        # Log Curiosities
        log_naive_dict = get_logs(naive_dict)
        log_free_dict = get_logs(free_dict)
        min_max = (log(min_max[0]), log(min_max[1]))
        
        ax = axs[13,i] if len(plot_dicts) > 1 else axs[13]
        awesome_plot(ax, log_naive_dict, "green", "log Naive")
        awesome_plot(ax, log_free_dict, "red", "log Free")
        ax.set_ylabel("log Curiosity")
        ax.legend()
        ax.set_title(plot_dict["title"] + "\nlog Possible Curiosities")
        
        ax = axs[14,i] if len(plot_dicts) > 1 else axs[14]
        awesome_plot(ax, log_naive_dict, "green", "log Naive", min_max)
        awesome_plot(ax, log_free_dict, "red", "log Free", min_max)
        ax.set_ylabel("log Curiosity")
        ax.legend()
        ax.set_title(plot_dict["title"] + "\nlog Possible Curiosities, shared min/max")
        
        
        
        print("\n{}: {}.".format(plot_dict["title"], duration()))

    
    
    # Done!
    fig.tight_layout(pad=1.0)
    plt.savefig("saved/plot.png", bbox_inches = "tight")
    plt.show()
    plt.close()