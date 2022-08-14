from RevisedNumericalSolver import get_full_state as runge_state
from TorchNumericalSolver import get_full_state
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import torch
from datetime import datetime
mpl.rcParams['animation.ffmpeg_path'] = r'C:\Users\Lazarus\Desktop\ThreeBodies\ffmpeg-5.0-essentials_build\bin\ffmpeg.exe'


# Function to determine whether gpu is available or not
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()


# Automatically puts data onto the default device
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)



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
    max_val = torch.tensor([100000000]).to(device)
    index = -1
    while i < max:
        if nearest_position(particle, state, data_set[i]).item() < max_val.item():
            index = i
            max_val = nearest_position(particle, state, data_set[i])

        i += 1
    #print(f"Time: {index*time_step}")
    return index


figure, ax = plt.subplots(2, 1)
top, bottom = ax
top.set_xlim(-1,1.5)
top.set_ylim(-2,2)
particle1, = top.plot([], [], color='r', label="Real First Star")
particle2, = top.plot([], [], color='g', label="Real Second Star")
particle3, = top.plot([], [], color='b', label="Real Third Star")
first_text = bottom.text(0.7, 0.85, "", fontsize = "xx-small", transform=ax[1].transAxes)
second_text = bottom.text(0.7, 0.78, "", fontsize = "xx-small", transform = ax[1].transAxes)
third_text = bottom.text(0.7, 0.71, "", fontsize = "xx-small", transform = ax[1].transAxes)
step_text = bottom.text(0.7, 0.64, "", fontsize = "xx-small", transform = ax[1].transAxes)
loss_values = []
#ax.legend(loc="upper left", fontsize=28)
top.legend(loc="upper left", fontsize=6)



def update(i, time_step, max_period, vec, m_1, m_2, m_3, optimizer, is_rand):
    input_vec = torch.cat((vec, torch.tensor([m_1,m_2,m_3])))

    data_set = runge_forward(input_vec, time_step, max_period)
    step = time_step
    step_text_val = f"dt: .{step} (Runge Kutta)"
    print(f"Max Period: {max_period}")

    #optimizer = torch.optim.Adam([input_vec], lr = lr)
    if len(loss_values) > 10:
        if loss_values[-1] == loss_values[-3]:
            print("Repeated")
            optimizer = torch.optim.SGD([vec], lr=.00001)
    #     else:
    #         optimizer = opt_func([vec], lr = lr)
    # else:
    #     optimizer = opt_func([vec], lr = lr)
    optimizer.zero_grad()

    #data_set = forward(input_vec)
    if i < 2 or i % 2 == 1:
               # data_set = forward(input_vec)
        first_index = nearest_position_state(1, data_set[0], data_set, 300, len(data_set), step)
        first_particle_state = data_set[first_index]
        second_index = nearest_position_state(2, data_set[0], data_set, 300, len(data_set), step)
        second_particle_state = data_set[second_index]
        third_index = nearest_position_state(3, data_set[0], data_set, 300, len(data_set), step)
        third_particle_state = data_set[third_index]
        loss = nearest_position(1, data_set[0], first_particle_state) + nearest_position(2, data_set[0],
                                                                                         second_particle_state) + nearest_position(
            3, data_set[0], third_particle_state)
        global period_index
        period_index = int((first_index + second_index + third_index) / 3)
        #global period_index
        #period_index = int((first_index + second_index + third_index) / 3)

    else:

        
        
        print(period_index)
        #data_set = forward(input_vec)
        first_index = nearest_position_state(1, data_set[0], data_set, period_index + 100, len(data_set), step)
        first_particle_state = data_set[first_index]
        second_index = nearest_position_state(2, data_set[0], data_set, period_index + 100, len(data_set), step)
        second_particle_state = data_set[second_index]
        third_index = nearest_position_state(3, data_set[0], data_set, period_index+100, len(data_set), step)
        third_particle_state = data_set[third_index]
        loss = nearest_position(1, data_set[0], first_particle_state) + nearest_position(2, data_set[0],
                                                                                         second_particle_state) + nearest_position(
            3, data_set[0], third_particle_state)


    print(" ")

    loss_values.append(loss.item())

    print(loss)

    loss.backward()
    print(vec.grad)
    # Updates input vector
    optimizer.step()
    print(input_vec)

    data_set = data_set.cpu().detach().numpy()

    particle1.set_data(data_set[:, 0], data_set[:, 1])
    particle2.set_data(data_set[:, 3], data_set[:, 4])
    particle3.set_data(data_set[:, 6], data_set[:, 7])

    first_text.set_text(f"First Particle Time: {round(first_index*step, 2)}")
    second_text.set_text(f"Second Particle Time: {round(second_index*step, 2)}")
    third_text.set_text(f"Third Particle Time: {round(third_index*step, 2)}")
    step_text.set_text(step_text_val)

    bottom.plot([x for x in range(len(loss_values))], loss_values, color="red")

    print(f"Epoch:{i}")
    print(" ")
    if is_rand:
        with open("randlog.txt", "a") as file:
            file.write(f"Epoch: {i}\nMax Period: {max_period}\nInput Vec: {input_vec}\nLoss: {loss}\nGradient: {vec.grad}\n\n")
    else:
        with open("log.txt", "a") as file:
            file.write(f"Epoch: {i}\nMax Period: {max_period}\nInput Vec: {input_vec}\nLoss: {loss}\nGradient: {vec.grad}\n\n")    
    return particle1, particle2, particle3, first_text, second_text, third_text







# now = str(datetime.now())
# video_title = now[:now.index(".")]
# video_title = video_title[:video_title.index(" ")]  + "-" + video_title[video_title.index(" ")+1:]
# video_title = video_title[:video_title.index(":")]  + "-" + video_title[video_title.index(":")+1:]
# video_title = video_title[:video_title.index(":")]  + "-" + video_title[video_title.index(":")+1:]
# print(video_title)
# print("NAdam: \n")
# ani1 = animation.FuncAnimation(figure, update, frames=num_epochs, fargs=(learning_rate, .002, len_stages, 10, vec, m_1, m_2, m_3, torch.optim.NAdam))
# ani1.save(f"D:\\Main\\PycharmProjects\\PeriodicThreeBodies\\Videos\\AfterJuly18\\a{video_title}.mp4", writer=writer)


# m_1 = 1
# m_2 = 1
# m_3 = 0.75
# v_1 = 0.4227625247
# v_2 = 0.2533646387
# vec = torch.tensor([-1,0,0,1,0,0,0,0,0,v_1, v_2, 0, v_1, v_2, 0, -2*v_1/m_3, -2*v_2/m_3, 0], requires_grad = True)
# vec = perturb(vec, .01)
# print("RAdam: \n")
# ani2 = animation.FuncAnimation(figure, update, frames=num_epochs, fargs=(learning_rate, .002, 10, vec, m_1, m_2, m_3, torch.optim.RAdam))
# ani2.save(f"D:\\Main\\PycharmProjects\\PeriodicThreeBodies\\Videos\\AfterJuly18\\a{video_title}.mp4", writer=writer)




def optimize(vec, m_1, m_2, m_3, lr=.0001, time_step = .002, num_epochs=20, max_period=10, opt_func = torch.optim.NAdam, video_folder="AfterJuly18"):
    with open("log.txt", "a") as file:
        file.write(f"\n\n\n{vec}, {m_1}, {m_2}, {m_3}, {opt_func}, {lr}, {time_step}")
    fp = int(num_epochs/10)
    if fp == 0:
        fp = 1
    writer = animation.FFMpegWriter(fps = fp)
    optimizer = opt_func([vec], lr=lr)
    now = str(datetime.now())
    video_title = now[:now.index(".")]
    video_title = video_title[:video_title.index(" ")] + "-" + video_title[video_title.index(" ") + 1:]
    video_title = video_title[:video_title.index(":")] + "-" + video_title[video_title.index(":") + 1:]
    video_title = video_title[:video_title.index(":")] + "-" + video_title[video_title.index(":") + 1:]
    a = int(vec[0].item()*1000)
    print(a)
    ani = animation.FuncAnimation(figure, update, frames=num_epochs, fargs=(time_step, max_period, vec, m_1, m_2, m_3, optimizer, False))
    ani.save(f"C:\\Users\\Lazarus\\Desktop\\ThreeBodies\\Videos\\{video_folder}\\{a}{video_title}.mp4",
              writer = writer)

def randoptimize(vec, m_1, m_2, m_3, lr=.0001, time_step = .002, num_epochs=20, max_period=10, opt_func = torch.optim.NAdam, video_folder="rand"):
    with open("randlog.txt", "a") as file:
        file.write(f"\n\n\n{vec}, {m_1}, {m_2}, {m_3}, {opt_func}, {lr}, {time_step}")
    fp = int(num_epochs/10)
    if fp == 0:
        fp = 1
    writer = animation.FFMpegWriter(fps = fp)
    optimizer = opt_func([vec], lr=lr)
    now = str(datetime.now())
    video_title = now[:now.index(".")]
    video_title = video_title[:video_title.index(" ")] + "-" + video_title[video_title.index(" ") + 1:]
    video_title = video_title[:video_title.index(":")] + "-" + video_title[video_title.index(":") + 1:]
    video_title = video_title[:video_title.index(":")] + "-" + video_title[video_title.index(":") + 1:]
  
    ani = animation.FuncAnimation(figure, update, frames=num_epochs, fargs=(time_step, max_period, vec, m_1, m_2, m_3, optimizer, True))
    ani.save(f"C:\\Users\\Lazarus\\Desktop\\ThreeBodies\\Videos\\{video_folder}\\a{video_title}.mp4",
              writer = writer)

  #  with open("mainoutput.txt", "a") as file:
  #      file.write("\n")
  #      file.write(f"{num_epochs}, {lr}, {opt_func}: \n")
  #      file.write(str(vec))


def losses(vec, m_1, m_2, m_3, lr=.0001, time_step = .002, num_epochs=100, max_period=10, opt_func = torch.optim.NAdam):
    initial_vec = vec
    optimizer = opt_func([vec], lr=lr)
    result = {}
    #loss_values = []
    i = 0
    while i < num_epochs:
        input_vec = torch.cat((vec, torch.tensor([m_1,m_2,m_3])))

        data_set = runge_forward(input_vec, time_step, max_period)
        step = time_step
        
        #print(f"Max Period: {max_period}")

        #optimizer = torch.optim.Adam([input_vec], lr = lr)
        if len(loss_values) > 10:
            if loss_values[-1] == loss_values[-3]:
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
        print(f"{i},{initial_vec},{input_vec},{loss}\n")
        #loss_values.append(loss.item())
        result[vec] = loss.item()
        with open("case1log.txt", "a") as file:
            file.write(f"{i},{initial_vec},{input_vec},{max_period},{loss}\n")

        
        
        #print(loss)

        loss.backward()
       
        # Updates input vector
        optimizer.step()
        
    
        #print(f"Epoch:{i}")
        #print(" ")

        i += 1

    with open("mainoutput.txt", "a") as file:
        file.write("\n")
        file.write(f"{num_epochs}, {lr}, {opt_func}: \n")
        file.write(str(vec))
    return result
    
