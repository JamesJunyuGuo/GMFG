import numpy  as np 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')

def get_policy(time,location,id,pop):
    pop_temp = pop[id]
    res = []
    for iter in range(len(pop_temp)):
        res.append(pop_temp[iter,time,location])
    
    return np.array(res)
    
    
def draw_evolution(time,location,pop):
    res_0 = get_policy(time,location,0,pop)
    res_1 = get_policy(time,location,1,pop)
    iteration = np.arange(len(res_1))
    fig,axs = plt.subplots(2,1,dpi=100, gridspec_kw={'hspace': 0.5})
    axs[0].plot(iteration,res_0[:,0],label='Left',alpha=0.2)
    axs[0].plot(iteration,res_0[:,1],label='Still',alpha=0.5)
    axs[0].plot(iteration,res_0[:,2],label='Right',alpha=0.8)
    axs[0].set_title("Population 1, Time {} ,Location {}".format(time,location))
    axs[0].legend(fontsize='small')
    
    
    axs[1].plot(iteration,res_1[:,0],label='Left',alpha=0.4)
    axs[1].plot(iteration,res_1[:,1],label='Still',alpha=0.6)
    axs[1].plot(iteration,res_1[:,2],label='Right',alpha=0.8)
    axs[1].set_title("Population 2, Time {} ,Location {}".format(time,location))
    axs[1].legend(fontsize='small')
    
    fig.savefig("./result/fig/time {};location {}.png".format(time,location))
    plt.show()



