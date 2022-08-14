import torch
from RevisedNumericalSolver import get_full_state as runge_state
import matplotlib.pyplot as plt


initial_vec =torch.tensor([-1.0018,  0.0289,  0.0000,  0.9649,  0.0147,  0.0000,  0.0129,  0.0171,
         0.0000,  0.4700,  0.2530,  0.0000,  0.4089,  0.2995,  0.0000, -1.1218,
        -0.6996,  0.0000, 1, 1, 0.75])

# Figure 1 final_vec = torch.tensor([-0.9821,  0.0296,  0.0000,  0.9654,  0.0286,  0.0000, -0.0122,  0.0099,
#         0.0000,  0.4239,  0.2520,  0.0000,  0.4193,  0.2714,  0.0000, -1.1214,
#        -0.6956,  0.0000,  1.0000,  1.0000,  0.7500])
final_vec = torch.tensor([-0.9815,  0.0307,  0.0000,  0.9658,  0.0285,  0.0000, -0.0130,  0.0095,
         0.0000,  0.4224,  0.2530,  0.0000,  0.4207,  0.2708,  0.0000, -1.1220,
        -0.6965,  0.0000,  1.0000,  1.0000,  0.7500])

final_vec = torch.tensor([-1.5725e-01,  8.4641e-04,  0.0000e+00,  9.9990e-01, -8.2659e-06,
         0.0000e+00, -3.1168e-03, -1.7701e-03,  0.0000e+00,  4.8847e-04,
         2.2894e-01,  0.0000e+00,  1.9608e-04,  4.2140e-01,  0.0000e+00,
         9.7078e-04, -1.2321e+00,  0.0000e+00,  6.1578e-01,  2.8282e-01,
         2.1215e-01])

#final_vec = torch.tensor([-0.9813,  0.0307,  0.0000,  0.9658,  0.0283,  0.0000, -0.0133,  0.0095,
#         0.0000,  0.4223,  0.2529,  0.0000,  0.4209,  0.2709,  0.0000, -1.1221,
#        -0.6967,  0.0000,  1.0000,  1.0000,  0.7500])
#final_vec = torch.tensor([-0.9816,  0.0303,  0.0000,  0.9660,  0.0291,  0.0000, -0.0134,  0.0091,
#         0.0000,  0.4216,  0.2531,  0.0000,  0.4209,  0.2694,  0.0000, -1.1221,
#        -0.6964,  0.0000,  1.0000,  1.0000,  0.7500])

#final_vec = torch.tensor([-1.5753e-01,  1.1131e-03,  0.0000e+00,  9.9963e-01, -2.8109e-04,
#         0.0000e+00, -2.8408e-03, -2.0326e-03,  0.0000e+00,  7.5900e-04,
#         2.2920e-01,  0.0000e+00, -7.7788e-05,  4.2113e-01,  0.0000e+00,
#         1.2385e-03, -1.2324e+00,  0.0000e+00,  6.1578e-01,  2.8282e-01,
#         2.1215e-01])

#initial_trajectory = runge_state(initial_vec, 0.002, 30).numpy()
#print(initial_trajectory)
#print(initial_trajectory.shape)

final = True
if final:
    final_trajectory = runge_state(final_vec, 0.001, 12)
else:
    final_trajectory = runge_state(initial_vec, 0.001, 13)
fig = plt.figure(figsize=(2,2))
particle1 = plt.plot(final_trajectory[:,0], final_trajectory[:,1], color='r', label="First Particle")
particle2 = plt.plot(final_trajectory[:,3], final_trajectory[:,4], color='g', label="Second Particle")
particle3 = plt.plot(final_trajectory[:,6], final_trajectory[:,7], color='b', label="Third Particle")
plt.legend(loc="upper right", fontsize=6)
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.show()
