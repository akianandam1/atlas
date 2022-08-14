from ModelVisuals import optimize
import torch
from RevisedNumericalSolver import torchstate
from shuffledcase1road import data
import time
import random
import sys

# Function to determine whether gpu is available or not
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()


# Perturbs input vector using normal distribution
# takes in float standard deviation
# Requires floats
def perturb(vec, std):
    return torch.tensor([torch.normal(mean=vec[0], std=torch.tensor(std)),
                         torch.normal(mean=vec[1], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[3], std=torch.tensor(std)),
                         torch.normal(mean=vec[4], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[6], std=torch.tensor(std)),
                         torch.normal(mean=vec[7], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[9], std=torch.tensor(std)),
                         torch.normal(mean=vec[10], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[12], std=torch.tensor(std)),
                         torch.normal(mean=vec[13], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[15], std=torch.tensor(std)),
                         torch.normal(mean=vec[16], std=torch.tensor(std)),
                         0.0,], requires_grad = True)


# Epoch ~6400 at .001 time length 15

# Compares two states and returns a numerical value rating how far apart the two states in the three
# body problem are to each other. Takes in two tensor states and returns tensor of value of distance rating
# The higher the score, the less similar they are
def nearest_position(particle, state1, state2):
    mse = torch.nn.L1Loss()
    if particle == 1:
        return mse(state1[:3], state2[:3]) + mse(state1[9:12], state2[9:12])
    elif particle == 2:
        return mse(state1[3:6], state2[3:6]) + mse(state1[12:15], state2[12:15])
    elif particle == 3:
        return mse(state1[6:9], state2[6:9]) + mse(state1[15:18], state2[15:18])
    else:
        print("bad input")


# Finds the most similar state to the initial position in a data set
def nearest_position_state(particle, state, data_set, min, max, time_step):
    i = min
    max_val = torch.tensor([100000000]).to(device)
    index = -1
    while i < max:
        if nearest_position(particle, state, data_set[i]).item() < max_val.item():
            index = i
            max_val = nearest_position(particle, state, data_set[i])

        i += 1
    #print(f"Time: {index*time_step}")
    return index

# beginning tensor([-1.0018,  0.0289,  0.0000,  0.9649,  0.0147,  0.0000,  0.0129,  0.0171,
#          0.0000,  0.4700,  0.2530,  0.0000,  0.4089,  0.2995,  0.0000, -1.1218,
#         -0.6996,  0.0000,  1.0000,  1.0000,  0.7500], grad_fn=<CatBackward0>)




def loss_values(identity, vec, m_1, m_2, m_3, lr, time_step, num_epochs, max_period, opt_func):
    initial_vec = vec
    optimizer = opt_func([vec], lr=lr)
    losses = []
    #result = {}
    #loss_values = []
    i = 0
    while i < num_epochs:
        input_vec = torch.cat((vec, torch.tensor([m_1,m_2,m_3])))

        data_set = torchstate(input_vec, time_step, max_period, "explicit_adams")
        
        
        

        #optimizer = torch.optim.Adam([input_vec], lr = lr)
        if len(losses) > 10:
            if losses[-1] == losses[-3]:
                #print("Repeated")
                optimizer = torch.optim.SGD([vec], lr=.00001)
            else:
                optimizer = opt_func([vec], lr=lr)
        #     else:
        #         optimizer = opt_func([vec], lr = lr)
        # else:
        #     optimizer = opt_func([vec], lr = lr)
        optimizer.zero_grad()

        #data_set = forward(input_vec)
        first_index = nearest_position_state(1, data_set[0], data_set, 300, len(data_set), step)
        first_particle_state = data_set[first_index]
        second_index = nearest_position_state(2, data_set[0], data_set, 300, len(data_set), step)
        second_particle_state = data_set[second_index]
        third_index = nearest_position_state(3, data_set[0], data_set, 300, len(data_set), step)
        third_particle_state = data_set[third_index]
        loss = nearest_position(1, data_set[0], first_particle_state) + nearest_position(2, data_set[0],
                                                                                         second_particle_state) + nearest_position(
            3, data_set[0], third_particle_state)

        #print(" ")
        
        print(f"{identity},{i},{loss}\n")
        losses.append(loss.item())
      
        with open("lossvalues\\case1roadloss.txt", "a") as file:
            file.write(f"{identity},{i},{loss}\n")

        
        
        #print(loss)

        loss.backward()
       
        # Updates input vector
        optimizer.step()
        
    
        #print(f"Epoch:{i}")
        #print(" ")

        i += 1


    #return result
    
def batch_losses(data, start, end):
    i = start
    while i <= end:
        m_1 = float(data[i][0])
        m_2 = float(data[i][1])
        m_3 = float(data[i][2])
        x_1 = float(data[i][3])
        v_1 = float(data[i][4])
        v_2 = float(data[i][5])
        T = float(data[i][6])
        vec = torch.tensor([x_1,0,0,1,0,0,0,0,0, 0, v_1, 0, 0, v_2, 0, 0, -(m_1*v_1 + m_2*v_2)/m_3, 0], requires_grad = True)
        vec = perturb(vec, .01)
        loss_values(i, vec, m_1, m_2, m_3, .0001, .002, 500, int(T+2), torch.optim.NAdam)

        i += 1
    


def get_loss(input_set):
    print("Begun")
    m_1 = float(input_set[0])
    m_2 = float(input_set[1])
    m_3 = float(input_set[2])
    x_1 = float(input_set[3])
    v_1 = float(input_set[4])
    v_2 = float(input_set[5])
    T = float(input_set[6])
    vec = torch.tensor([x_1,0,0,1,0,0,0,0,0, 0, v_1, 0, 0, v_2, 0, 0, -(m_1*v_1 + m_2*v_2)/m_3, 0], requires_grad = True)
    #print(vec)
    vec = perturb(vec, .01)
    return loss_values(vec, m_1, m_2, m_3, lr = .0001, time_step = .001, num_epochs = 90, max_period=int(T+2))




if __name__ == "__main__":
    
        
    start1 = time.time()
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    batch_losses(data, start, end)
      
 


    
    
    end1 = time.time()
    print(f"This process took {(end1-start1)/3600} hours")
    







