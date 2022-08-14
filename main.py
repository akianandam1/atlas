from ModelVisuals import optimize
import torch
from RevisedNumericalSolver import torchstate
#from TorchNumericalSolver import get_full_state
from shuffledcase1road import data
#import torch.multiprocessing as mp
#from contextlib import closing
import time
import random
import sys
#import gc
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

def forward(input_vec, time_step, time_length):
    return get_full_state(input_vec, time_step, time_length)
def runge_forward(input_vec, time_step, time_length):
    return runge_state(input_vec, time_step, time_length)
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
    max_val = torch.tensor([100000000])
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


def make_video(input_set):
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
    optimize(vec, m_1, m_2, m_3, lr = .0001, time_step = .001, num_epochs = 90, max_period=int(T+2), video_folder = "Case1")



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
        step = time_step
        
        #print(f"Max Period: {max_period}")

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

def parrallelize(dataset):
    with closing(Pool(30)) as p:
        result = p.imap_unordered(get_loss, dataset, 16)
    return result 


if __name__ == "__main__":
    #print(data)
    #shuffled = random.sample(data, 100)
    #print(shuffled)
    #with open("shuffledcase1road.py", "w") as file:
    #    file.write(f"data = {shuffled}")
        
    start1 = time.time()
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    batch_losses(data, start, end)
    #print(sys.argv)
    #a = int(sys.argv[1])
    #get_loss(data[a])
    

         
   # m_1 = 1#6.1578e-01#6.1578e-01
   # m_2 = 1#2.8282e-01#2.8282e-01
   # m_3 = 0.75#2.1215e-01#2.1215e-01
 #   vec = torch.tensor([-0.9828,  0.0287,  0.0000,  0.9644,  0.0275,  0.0000, -0.0105,  0.0113,
 #        0.0000,  0.4277,  0.2505,  0.0000,  0.4174,  0.2736,  0.0000, -1.1203,
 #       -0.6951,  0.0000], requires_grad = True)
    #vec = torch.tensor([-1.0018, 0.0289, 0.0000, 0.9649, 0.0147, 0.0000, 0.0129, 0.0171, 0.0000, 0.4700, 0.2530, 0.0000, 0.4089, 0.2995, 0.0000, -1.1218, -0.6996, 0.0000], requires_grad = True)

    #vec = torch.tensor([-1.5753e-01,  1.1131e-03,  0.0000e+00,  9.9963e-01, -2.8109e-04,
    #     0.0000e+00, -2.8408e-03, -2.0326e-03,  0.0000e+00,  7.5900e-04,
    #     2.2920e-01,  0.0000e+00, -7.7788e-05,  4.2113e-01,  0.0000e+00,
    #     1.2385e-03, -1.2324e+00,  0.0000e+00], requires_grad = True)
   # vec = torch.tensor([-1.5745e-01,  1.0118e-03,  0.0000e+00,  9.9953e-01, -1.7825e-04,
   #      0.0000e+00, -2.8845e-03, -1.9329e-03,  0.0000e+00,  6.5842e-04,
   #      2.2930e-01,  0.0000e+00,  2.5311e-05,  4.2103e-01,  0.0000e+00,
   #      1.1374e-03, -1.2323e+00,  0.0000e+00])


 #   vec = torch.torch.tensor([-0.9815,  0.0307,  0.0000,  0.9658,  0.0285,  0.0000, -0.0130,  0.0095,
 #        0.0000,  0.4224,  0.2530,  0.0000,  0.4207,  0.2708,  0.0000, -1.1220,
 #       -0.6965,  0.0000], requires_grad = True)
 #   vec = torch.tensor([-0.9815,  0.0307,  0.0000,  0.9658,  0.0285,  0.0000, -0.0130,  0.0095,
 #        0.0000,  0.4224,  0.2530,  0.0000,  0.4207,  0.2708,  0.0000, -1.1220,
 #       -0.6965,  0.0000], requires_grad = True)
 #   T = 17
 #   optimize(vec, m_1, m_2, m_3, lr=.0001, time_step=.001, num_epochs = 500, max_period=T)
    
    
   # pool.map(get_loss, data[:14])
    
 #   processes = []
 #   for vec in data[:14]:
        #print(vec)
 #       p = Process(target=get_loss, args=(vec,))
 #       p.start()
 #       processes.append(p)
 #   for process in processes:
 #       process.join()
   # p = Process(target=get_loss, args=data[:14])
   # p.start()
   # p.join()
  #  with Pool() as pool:
   #     result = pool.map(get_loss, data[:8])
     
    
  #  with Pool() as pool:
   #     result1 = pool.map(get_loss, data[:8])
    #with open("case1losses.py", "w") as file:
 #       file.write(f"losses1 = {result1} \n")
  #  print("First Done \n" *5)
    
  # with Pool() as pool:
    #result2 = pool.map(get_loss, data[8:16])
  #  with open("case1losses.py", "w") as file:
   #     file.write(f"losses2 = {result2} \n")
  #  print("Second Done \n" *5)
    
   # with Pool() as pool:
   #     result3 = pool.map(get_loss, data[16:24])
   # with open("case1losses.py", "w") as file:
   #     file.write(f"losses3 = {result3} \n")
   # print("Third Done \n" *5)
    
   # with Pool() as pool:
   #     result4 = pool.map(get_loss, data[24:32])
   # with open("case1losses.py", "w") as file:
   #     file.write(f"losses4 = {result4} \n")
   # print("Fourth Done \n" *5)

    
    
    
    end1 = time.time()
    print(f"This process took {(end1-start1)/3600} hours")
    
# m_1 = 1
# m_2 = 1
# m_3 = 0.75
#
# v_1 = 0.2827020949
# v_2 = 0.3272089716
# T = 10.9633031497
# vec = torch.tensor([-1,0,0,1,0,0,0,0,0,v_1, v_2, 0, v_1, v_2, 0, -2*v_1/m_3, -2*v_2/m_3, 0], requires_grad = True)
# vec = torch.tensor([-1,0,0, 1,0,0, 0,0,0, v_1,v_2,0, v_1,v_2,0, -2*v_1,-2*v_2,0], requires_grad=True)

# FIGURE 1 vec = torch.tensor([-1.0018,  0.0289,  0.0000,  0.9649,  0.0147,  0.0000,  0.0129,  0.0171,
#          0.0000,  0.4700,  0.2530,  0.0000,  0.4089,  0.2995,  0.0000, -1.1218,
#         -0.6996,  0.0000], requires_grad = True)


# vec = torch.tensor([-0.9937,  0.0193,  0.0000,  0.9634,  0.0237,  0.0000,  0.0061,  0.0221,
#          0.0000,  0.4581,  0.2431,  0.0000,  0.4085,  0.2965,  0.0000, -1.1173,
#         -0.6903,  0.0000], requires_grad = True)

# vec = perturb(vec, .01)
#optimize(vec, m_1, m_2, m_3, lr = .0001, num_epochs = 40, max_period=int(T+2), opt_func=torch.optim.Adagrad)
# optimize(vec, m_1, m_2, m_3, lr = .00001, num_epochs = 90, max_period=int(T+2), opt_func=torch.optim.SGD)

# vec = torch.tensor([-0.9957,  0.0225,  0.0000,  0.9649,  0.0178,  0.0000,  0.0070,  0.0205,
#          0.0000,  0.4640,  0.2438,  0.0000,  0.4100,  0.2989,  0.0000, -1.1197,
#         -0.6929,  0.0000], requires_grad = True)
#
#
# vec = vec = perturb(vec, .02)


