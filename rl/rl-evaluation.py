from model import *

import torch
import numpy as np

from stable_baselines3.common.policies import obs_as_tensor



def predict_proba(model, state):
    obs = model.policy.obs_to_tensor(state)[0]
    #obs = obs_as_tensor(state, model.policy.device)
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs

    probs_np = probs.detach().numpy()
    #print(probs_np[0].std(), probs_np[0])
    return probs_np[0]
def model_p(model, state, env):
    probs = predict_proba(model, state)
    mask = env.action_masks()
    idx = np.argmax(probs[mask])

    pred = indexes[mask][idx]
    actions.append(pred)
    print(pred, np.max(probs[env.action_masks()]), np.std(probs))
    return pred
def model_p_m(model, state, env):
    obs = model.policy.obs_to_tensor(state)[0]
    dis = model.policy.get_distribution(obs)
    dis.apply_masking(env.action_masks())
    print(dis.probs)

def random_m(env):

    pred1 = np.random.choice(np.where(env.action_masks()[0] == True)[0])
    pred2 = np.random.choice(np.where(env.action_masks()[1] == True)[0])
    return np.array([pred1, pred2])
def min_dist(state):
    distance = state["distmap"]
    
    idx = np.argmin(distance[distance > 0])
    return indexes[distance > 0][idx]


import re


path = "C:/Users/jon39/OneDrive/NTNU/5. år/Master/thesis-NTNU-master/figures/ex4/"

def dict_to_latex(dic):
    for k in list(dic.keys()):
        if k in ["Model", "Mean Response Time", "Mean Reward", "Haversine Fraction", "Iterations"]:
            dic["\\textbf{"+k+"}"] = dic[k]
        del dic[k]
    df = pd.DataFrame.from_dict(dic)
    tex_content = df[list(dic.keys())].to_latex(index=False, escape=False)

    import re # Use regular expression to search for column declaration section of LaTeX table
    #tex_content = df.to_latex(index=True, escape=False) #Or whatever options you want/need

    re_borders = re.compile(r"begin\{tabular\}\{([^\}]+)\}")
    borders = re_borders.findall(tex_content)[0]
    borders = borders.replace("r", "l")
    borders = '|'+'|'.join(list(borders))+'|'
    tex_content = re_borders.sub("begin{tabular}{" + borders + "}", tex_content)

    res = ["\\begin{table}[]", "\centering", "\\resizebox{\\textwidth}{!}{%"]
    for f in tex_content.split("\n"):
        if not ("rule" in f):
            res.append(f)
            #if "begin{tabular}" in f:
            #    res.append("\\hline")
    res.append("}")
    res.append("\end{table}")
    tex_content = "\n".join(res)
    print(tex_content)
    with open(path+"res.tex", "w+") as f:
        f.write(tex_content)
def create_plots_reWaTime(df):
    fig, axs = plt.subplots(2,2)
    ax1 = axs[0,0]
    ax2 = axs[0,1]
    ax3 = axs[1,0]
    ax4 = axs[1,1]


    fig.set_size_inches(w=5.11911, h=5.11911)
    x = np.linspace(0,60,1000)

    bins = 100
    xli = (0,90)

    for i, r in df.iterrows(): # RESPONSETIME

        rTime = np.array(r["Response Times"])
        rTime = rTime[rTime > 0]

        #ax1.hist(rTime, bins=bins, label=r["Model"], alpha=0.5)
        ax1.hist(rTime, bins=bins, label=r["Model"], histtype='step', stacked=True, fill=False)
        #sns.histplot(rTime, bins=bins, label=r["Model"], alpha=0.5, ax=ax1)
        
        ax1.set_title("Response time histogram")
        ax1.set_xlabel("Response time (m)")
        ax1.set_xlim(xli)
        vals = []
        
        for xi in x:
            vals.append(len(rTime[rTime < xi])/len(rTime))
        ax2.plot(x, vals, label=r["Model"])
        ax2.set_xlim((5,30))
        ax2.set_title("Response time cdf")
        ax2.set_xlabel("Response time (m)")
        
        
        rew = np.array(r["Rewards"])
        rew = -rew[rew < 0]
        #rew = -rew
        #ax3.hist(rew, bins=bins, label=r["Model"], alpha=0.5)
        ax3.hist(rew, bins=bins, label=r["Model"], histtype='step', stacked=True, fill=False)
        #sns.histplot(rew, bins=bins, label=r["Model"], alpha=0.5, ax=ax3)

        ax3.set_title("Waittime histogram")
        ax3.set_xlim(xli)
        ax3.set_xlabel("Waittime (m)")
        #ax3.set
        vals = []
        for xi in x:
            vals.append(len(rew[rew < xi])/len(rew))
        ax4.plot(x, vals, label=r["Model"])
        ax4.set_xlim((5,30))
        ax4.set_title("Wait time cdf")
        ax4.set_xlabel("Wait time (m)")
    #ax1.legend()
    #ax2.legend()
    #ax3.legend()
    #ax4.legend()
    #plt.legend(loc='lower center', ncol=1)
    plt.subplots_adjust(wspace=0.256, hspace=0.445)
    plt.savefig(path+"ex1.png", dpi=100)
    plt.show()
def create_plots_surResTime(df):
    fig, axs = plt.subplots(2,2)
    ax1 = axs[0,0]
    ax2 = axs[0,1]
    ax3 = axs[1,0]
    ax4 = axs[1,1]


    fig.set_size_inches(w=5.11911, h=5.11911)
    x = np.linspace(0,60,1000)
    x2 = np.linspace(0,1, 1000)
    bins = 100
    xli = (0,90)

    for i, r in df.iterrows(): # RESPONSETIME

        rTime = np.array(r["Response Times"])
        rTime = rTime[rTime > 0]

        #ax1.hist(rTime, bins=bins, label=r["Model"], alpha=0.5)
        ax1.hist(rTime, bins=bins, label=r["Model"], histtype='step', stacked=True, fill=False)
        #sns.histplot(rTime, bins=bins, label=r["Model"], alpha=0.5, ax=ax1)
        
        ax1.set_title("Response time histogram")
        ax1.set_xlabel("Response time (m)")
        ax1.set_xlim(xli)
        vals = []
        
        for xi in x:
            vals.append(len(rTime[rTime < xi])/len(rTime))
        ax2.plot(x, vals, label=r["Model"])
        ax2.set_xlim((5,30))
        ax2.set_title("Response time cdf")
        ax2.set_xlabel("Response time (m)")
        
        
        rew = np.array(r["Rewards"])
        #rew = -rew[rew < 0]
        #rew = -rew
        #ax3.hist(rew, bins=bins, label=r["Model"], alpha=0.5)
        ax3.hist(rew, label=r["Model"], histtype='step', stacked=True, fill=False)
        #sns.histplot(rew, bins=bins, label=r["Model"], alpha=0.5, ax=ax3)

        ax3.set_title("Survivability histogram")
        #ax3.set_xlim((0.2,0.8))
        ax3.set_xlabel("Survivability fraction")
        #ax3.set
        vals = []
        for xi in x2:
            vals.append(len(rew[rew > xi])/len(rew))
        ax4.plot(x2, vals, label=r["Model"])
        #ax4.set_xlim((5,30))
        ax4.set_title("Survivability inverse cdf")
        ax4.set_xlabel("Survivability fraction")
    #ax1.legend()
    #ax2.legend()
    #ax3.legend()
    #ax4.legend()
    #plt.legend(loc='lower center', ncol=1)
    plt.subplots_adjust(wspace=0.256, hspace=0.445)
    plt.savefig(path+"ex1.png", dpi=100)
    plt.show()


#df[["Model", "Mean Response Time", "Mean Reward", "Min Policy Fraction", "Incidents attended", "Iterations"]].to_csv("results.csv")
#plt.rcParams["figure.figsize"] = (11,11)

import seaborn as sns

if __name__ == "__main__":
    
    TRAINING = False
    #model, env = get_model_env(training=False)
    env = dispatchEnv(lamb=0.21, LaLo=[59.910986, 10.752496], dist=5, fromData=True, waitTimes=False, training=False)
    env = ActionMasker(env, mask_fn)
    print(env.fromData)


    model = proposed_model(env)
    model.set_parameters("./ex1/rl_model_7890000_steps.zip")
    #model = model.load("./logs/rl_model_4110000_steps.zip")
    #model.set_parameters("./logs/rl_model_4110000_steps.zip")
    #model = 
    actions = []




    indexes = np.arange(env.nAmbulance)

    #env.iters = 1000

    ITERATIONS = 100#50000

    EPOCHS = 1
    ListACTIONS = False
    #SAVE = False
    results = {"Model" : [], "Mean Response Time" : [], "Mean Reward" : [], "Haversine Fraction" : [], "Incidents attended": [], "Iterations" : [], "Response Times" : [], "Rewards" : []}
    model_names = ["RL", "Haversine", "Euclidean", "Random"]
    for i in range(0,4):
        rewards = []
        responseTimes = []
        iter = 0
        min_dist_count = 0
        if not env.fromData:
            state = env.reset()
            #print(np.min(state["distmap"]))
            term = False
            while not term:
                iter += 1
                print(iter)

                if i == 0:
                    pred = model_p(model, state, env)
                elif i == 1:
                    pred = min_dist_info(env)[0]
                elif i == 2:
                    pred = min_dist_info_euc(env)[0]
                elif i == 3:
                    #exit()
                    print("random")
                    pred = random(env)
                choice, indices = min_dist_info(env)
                if not ListACTIONS:
                    if pred in indices:
                        min_dist_count += 1
                else:
                    print(pred, indices)
                    print(np.where(pred == indices)[0].shape[0])
                    if np.where(pred == indices)[0].shape[0] > 0:
                        min_dist_count += 1
                    else:
                        print("not best")
                state, reward, term = env.step(pred)[:3]
                rewards.append(reward)

                if iter > ITERATIONS:
                    term = True
            responseTimes.extend(env.responseTime)

        else:
            env.dataReader.reset_iter()
            state = env.reset()

            env.dataReader.epoch = 0
            while env.dataReader.epoch < EPOCHS:#env.dataReader.iter < 1000: #
                e = env.dataReader.epoch
                iter += 1
                #print(iter)
                if i == 0:
                    print(state)
                    acM = np.concatenate([state["ambulance"] > 0, state["incident"] > 0], axis=0)
                    pred = model.predict(state, deterministic=True, action_masks=mask_fn(env))[0]
                    #pred = model_p(model, state, env)
                elif i == 1:
                    pred = min_dist_info(env)[0]
                elif i == 2:
                    pred = min_dist_info_euc(env)[0]
                elif i == 3:
                    #exit()
                    
                    pred = random(env)
                choice, indices = min_dist_info(env)

                if not ListACTIONS:
                    if pred in indices:
                        min_dist_count += 1
                else:
                    if np.where(pred == indices)[0].shape[0] > 1:
                        min_dist_count += 1
                
                dt = env.dataReader
                print(dt.iter/len(dt.test.index), dt.epoch, min_dist_count, model_names[i])
                #print(env.action_masks())
                state, reward, term = env.step(pred)[:3]
                rewards.append(reward)

                if env.dataReader.iter == len(env.dataReader.test.index)-1:
                    #print(reset)
                    print("term")
                    responseTimes.extend(env.responseTime)
                    env.dataReader.iter = 0
                    env.dataReader.epoch += 1
                    #env.dataReader.reset_iter()
                    state = env.reset()
                    
                
            print("finished")
            #responseTimes.extend(env.responseTime)
        responseTimes = np.array(responseTimes)
        #rT = responseTimes[responseTimes > 0]
        results["Model"].append(model_names[i])
        results["Mean Response Time"].append(np.round(np.mean(responseTimes), 2))
        results["Mean Reward"].append(np.round(np.mean(rewards), 2))
        results["Haversine Fraction"].append(np.round(min_dist_count/iter, 2))
        results["Incidents attended"].append(len(responseTimes))
        results["Iterations"].append(iter)
        results["Response Times"].append(responseTimes)
        results["Rewards"].append(rewards)
    df = pd.DataFrame.from_dict(results)
    print(df)
    print("done")

    df.to_csv("out.csv")
    dict_to_latex(results)
    create_plots_reWaTime(df)
    
    u,c = np.unique(np.array(actions), return_counts=True)
    c = c/c.sum()
    plt.bar(u,c)
    plt.show()
