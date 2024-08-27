import json
from attention import Attention
import torch
import numpy as np
import traceback

np.set_printoptions(
    suppress=True,
    precision=4)

autograder_version = '3.0.1'
print("Autograder version: " + str(autograder_version))

input_dim = 5
seq_len = 10
batch_size = 1
output_dim = 5
d_q, d_k, d_v = 6, 6, 8

linear = torch.nn.Linear(d_v, output_dim)
criterion = torch.nn.CrossEntropyLoss()

TEST_attn_out = False
TEST_Q = False
TEST_K = False
TEST_V = False

TEST_dX = False
TEST_dV = False
TEST_dQ = False
TEST_dK = False

input = torch.tensor([[[-0.4191,  0.2760, -0.4906, -0.5526, -0.4345],
         [ 0.0772,  0.4980,  0.7445,  0.7883, -0.7991],
         [ 0.9661, -0.0473, -0.4255,  0.6711,  0.7136],
         [ 0.9133, -0.8818, -0.0430,  1.4497,  1.2950],
         [-0.0084,  1.5354, -0.0212,  1.2331, -1.1912],
         [-0.3051, -0.2506,  1.1343,  0.6214,  0.7300],
         [ 0.2802,  0.9361,  0.6056,  0.7327, -0.3326],
         [ 1.1226, -0.5197,  1.1643,  0.0484,  0.4477],
         [-0.2408, -0.3354,  0.2063,  1.1830, -0.9840],
         [-0.6865,  0.4439,  0.9313, -0.9110, -0.5126]],

        [[-0.0277,  0.5560, -0.2319,  1.4549, -0.5708],
         [-0.0707,  1.5911,  0.1544, -0.3544,  0.6401],
         [ 1.0143,  1.1292, -0.7290, -0.8952,  0.5538],
         [-1.4917,  0.4074, -0.2121,  1.7407, -0.4631],
         [ 0.9429,  1.3389,  0.8369,  0.4890, -0.6282],
         [-1.9038, -0.5821, -0.6364,  0.1368,  0.0770],
         [-1.6481,  0.0237,  0.2376,  0.1856,  1.6033],
         [ 1.1987, -1.4250,  0.9229,  2.3339,  1.5564],
         [-0.7571, -0.7816, -0.1630,  0.8074,  0.1243],
         [-1.3402, -0.7351, -0.4460,  1.0893, -1.1602]]])

W_query = torch.tensor([[0.1018, 0.8583, 0.6193, 0.8464, 0.4979, 0.6094],
        [0.8621, 0.2623, 0.3299, 0.2587, 0.9636, 0.8440],
        [0.5076, 0.4143, 0.0533, 0.5243, 0.0554, 0.5382],
        [0.2371, 0.5578, 0.4973, 0.7064, 0.6009, 0.6661],
        [0.0501, 0.7779, 0.1816, 0.7841, 0.3141, 0.0153]])

W_key = torch.tensor([[0.4675, 0.2260, 0.7307, 0.9349, 0.3413, 0.3039],
        [0.7197, 0.4092, 0.1292, 0.2489, 0.0929, 0.1403],
        [0.3257, 0.8796, 0.8690, 0.5451, 0.7872, 0.5148],
        [0.3967, 0.1984, 0.6724, 0.2447, 0.9245, 0.4676],
        [0.9716, 0.2081, 0.1205, 0.6515, 0.0321, 0.4499]])

W_value = torch.tensor([[0.1558, 0.1798, 0.1144, 0.1419, 0.9976, 0.0988, 0.6137, 0.3179],
        [0.4983, 0.8060, 0.4986, 0.4794, 0.2355, 0.3697, 0.0353, 0.3179],
        [0.1950, 0.1581, 0.2933, 0.7586, 0.4633, 0.3789, 0.9387, 0.6037],
        [0.3445, 0.7518, 0.1842, 0.3734, 0.5822, 0.4380, 0.9121, 0.3605],
        [0.7155, 0.8791, 0.0540, 0.7415, 0.1140, 0.6327, 0.1891, 0.5342]])

att_out = torch.tensor([[[-0.5247, -0.7278, -0.1794, -0.8278, -0.9516, -0.6422, -1.2942,
          -0.7730],
         [ 0.0692,  0.3574,  0.5344,  0.4395,  0.8010,  0.2590,  1.1818,
           0.4175],
         [ 0.4071,  0.7643,  0.3669,  0.5302,  1.0366,  0.4631,  1.1332,
           0.5554],
         [ 0.9204,  1.4199,  0.0962,  0.9943,  1.4971,  0.9944,  1.7964,
           1.0179],
         [ 0.9483,  1.4802,  0.1099,  0.9980,  1.5129,  1.0137,  1.7998,
           1.0214],
         [ 0.7834,  1.2206,  0.2351,  0.9787,  1.1591,  0.8807,  1.5394,
           0.9123],
         [ 0.8769,  1.3491,  0.2421,  1.0877,  1.2913,  0.9791,  1.6973,
           1.0176],
         [ 0.6859,  0.9469,  0.2864,  1.0760,  1.3467,  0.8179,  1.6970,
           1.0239],
         [ 0.3958,  0.7109,  0.3089,  0.5925,  0.8822,  0.4938,  1.1384,
           0.5716],
         [-0.1587, -0.1449,  0.1045, -0.1518, -0.1659, -0.1467, -0.1610,
          -0.1590]],

        [[ 0.3203,  0.9985,  0.4432,  0.2067,  0.7779,  0.3911,  1.0040,
           0.2474],
         [ 0.7940,  1.3374,  0.6473,  0.7822,  0.4713,  0.6764,  0.4038,
           0.5584],
         [ 0.7081,  1.2050,  0.5695,  0.6126,  0.5157,  0.5623,  0.3102,
           0.4615],
         [ 0.5216,  1.0661,  0.4436,  0.3984,  0.1988,  0.4640,  0.2762,
           0.2512],
         [ 0.7049,  1.2057,  1.0380,  1.0952,  1.7324,  0.7158,  1.6125,
           1.0246],
         [-0.5987, -0.7242, -0.6552, -0.9122, -2.2214, -0.5264, -1.6280,
          -1.0723],
         [ 0.3679,  0.6452,  0.1453,  0.2559, -0.5017,  0.3040, -0.3414,
           0.0343],
         [ 1.5682,  2.3278,  0.2170,  2.2050,  2.8171,  1.9398,  3.9591,
           2.1504],
         [-0.0527,  0.0776, -0.2172, -0.2285, -1.1254, -0.0224, -0.7024,
          -0.4007],
         [-0.7038, -0.7681, -0.5837, -0.9764, -1.8490, -0.5698, -1.2204,
          -1.0415]]])

K = torch.tensor([[[-0.7985, -0.6134, -1.1208, -1.0089, -1.0284, -0.7951],
         [ 0.1733,  0.8662,  1.2015,  0.2743,  1.3617,  0.4856],
         [ 1.2386,  0.1064,  0.8673,  1.2886,  0.6338,  0.7028],
         [ 1.6117,  0.3650,  1.6469,  1.8094,  1.5778,  1.3922],
         [ 0.4260,  0.6045,  0.8594, -0.1115,  1.2247,  0.2425],
         [ 1.0023,  1.1015,  1.2362,  0.8984,  1.3634,  1.0750],
         [ 0.9695,  1.0552,  1.3046,  0.7877,  1.3260,  0.7212],
         [ 0.9842,  1.1679,  1.8514,  1.8584,  1.3105,  1.0916],
         [-0.7735,  0.0198,  0.6368, -0.5478,  1.1110,  0.0964],
         [-0.5576,  0.5582, -0.3093, -0.5805, -0.3186, -0.3235]],

        [[ 0.3343,  0.1872,  0.7596, -0.0298,  1.1863,  0.3736],
         [ 1.6437,  0.8338,  0.1270,  0.7445, -0.0618,  0.4035],
         [ 1.2323, -0.0124, -0.2816,  0.9737, -0.9325, -0.0780],
         [-0.2325, -0.1079, -0.1070, -1.2846,  0.9560,  0.1002],
         [ 1.2606,  1.4633,  1.8424,  1.3814,  1.5369,  0.8512],
         [-1.3871, -1.1850, -1.9181, -2.1881, -1.0759, -0.8892],
         [ 0.9555,  0.2168, -0.6767, -0.3154, -0.1502,  0.4330],
         [ 2.2736,  1.2866,  3.2506,  2.8542,  3.2108,  2.4310],
         [-0.5284, -0.4482, -0.2380, -0.7127,  0.2911,  0.0098],
         [-1.9959, -1.0212, -0.8692, -2.1684,  0.0929, -0.7527]]])

Q = torch.tensor([[[-0.2066, -1.1368, -0.5483, -1.2716, -0.4384, -0.6612],
         [ 0.9619,  0.3234,  0.4987,  0.5148,  0.7822,  1.3809],
         [ 0.0365,  1.5699,  1.0233,  1.6160,  1.0392,  0.7777],
         [-0.2803,  2.3507,  1.2284,  2.5619,  0.8805,  0.7746],
         [ 1.5447,  0.1480,  0.8972,  0.3160,  1.8408,  2.0824],
         [ 0.5126,  1.0568,  0.2304,  1.2831,  0.2722,  0.6381],
         [ 1.3000,  0.8869,  0.8186,  1.0536,  1.4108,  1.7697],
         [ 0.2912,  1.6848,  0.6912,  1.8114,  0.2924,  0.9112],
         [ 0.0223, -0.3148,  0.1608, -0.1183, -0.0299,  0.4541],
         [ 0.5437, -0.9938, -0.7752, -1.0234, -0.5709, -0.1571]],

        [[ 0.6752,  0.3935,  0.7738,  0.5790,  1.2040,  1.2879],
         [ 1.3909,  0.7209,  0.4294,  0.6843,  1.4946,  1.1566],
         [ 0.5222,  0.7962,  0.6173,  0.5702,  1.1887,  0.5910],
         [ 0.4812, -0.6506, -0.0192, -0.4018,  0.5385,  0.4729],
         [ 1.7595,  1.2913,  1.1994,  1.4360,  1.9024,  2.4711],
         [-0.9824, -1.9141, -1.3230, -1.9385, -1.4376, -1.9017],
         [ 0.0976,  0.0409, -0.6168,  0.1241, -0.1694, -0.7084],
         [-0.0065,  3.5499,  1.7646,  3.9990,  1.1662,  1.6028],
         [-0.6359, -0.3753, -0.3114, -0.2606, -0.6149, -0.6691],
         [-0.7964, -1.8228, -0.7653, -1.6985, -1.1103, -0.9694]]])

V = torch.tensor([[[-0.5247, -0.7278, -0.1794, -0.8278, -0.9516, -0.6422, -1.2942,
          -0.7730],
         [ 0.1052,  0.4231,  0.5776,  0.5162,  0.9071,  0.3136,  1.3317,
           0.4895],
         [ 0.7857,  1.2002,  0.1242,  0.5714,  1.2276,  0.6622,  0.9389,
           0.6583],
         [ 1.1204,  1.6750, -0.0110,  1.1758,  1.6752,  1.2022,  2.0562,
           1.1985],
         [ 0.3321,  1.1126,  0.9212,  0.2959,  0.9256,  0.3453,  0.9286,
           0.2807],
         [ 0.7851,  1.0313,  0.3267,  1.4703,  0.6071,  1.0410,  1.5735,
           1.1221],
         [ 0.6426,  1.1591,  0.7934,  0.9748,  1.1692,  0.7137,  1.3789,
           0.8386],
         [ 0.4801,  0.3969,  0.2439,  1.1434,  1.6161,  0.6644,  1.8923,
           1.1512],
         [-0.4609, -0.2567,  0.0305, -0.3264,  0.3530, -0.1740,  0.9270,
          -0.1579],
         [-0.3847, -0.7539,  0.2206,  0.1016, -0.7377, -0.2742, -0.4593,
          -0.1171]],

        [[ 0.3203,  0.9985,  0.4432,  0.2067,  0.7779,  0.3911,  1.0040,
           0.2474],
         [ 1.1478,  1.5904,  0.7998,  1.2121,  0.2424,  0.8895, -0.0445,
           0.7907],
         [ 0.6664,  0.7912,  0.3302,  0.2087,  0.4821,  0.1997, -0.7338,
           0.2144],
         [ 0.1974,  0.9282,  0.2659,  0.1292, -0.5298,  0.3924,  0.4000,
          -0.0927],
         [ 0.6963,  1.1964,  1.0771,  1.1273,  1.8568,  0.7220,  1.7387,
           1.0712],
         [-0.6086, -0.7416, -0.6653, -0.9238, -2.2428, -0.5358, -1.6469,
          -1.0839],
         [ 1.0123,  1.3093,  0.0137,  1.2158, -1.2378,  1.0316, -0.3150,
           0.5505],
         [ 1.5742,  2.3356,  0.2111,  2.2126,  2.8240,  1.9483,  3.9748,
           2.1580],
         [-0.1722, -0.0756, -0.3687, -0.2121, -0.5307,  0.0068,  0.1148,
          -0.2301],
         [-1.1170, -1.1050, -0.5127, -1.3345, -1.2148, -0.8300, -0.4929,
          -1.1561]]])

dLXnew = torch.tensor([[[ 0.0021, -0.0021,  0.0029, -0.0134,  0.0030, -0.0133,  0.0036,
          -0.0059],
         [ 0.0073,  0.0071, -0.0039,  0.0107, -0.0094,  0.0124, -0.0069,
           0.0097],
         [-0.0108,  0.0023, -0.0148,  0.0136,  0.0045,  0.0110, -0.0029,
           0.0136],
         [ 0.0088,  0.0079, -0.0052,  0.0121, -0.0081,  0.0144, -0.0078,
           0.0133],
         [-0.0099,  0.0028, -0.0157,  0.0146,  0.0054,  0.0123, -0.0035,
           0.0160],
         [ 0.0046, -0.0009,  0.0009, -0.0107,  0.0061, -0.0096,  0.0020,
           0.0006],
         [-0.0102,  0.0028, -0.0157,  0.0149,  0.0049,  0.0125, -0.0035,
           0.0156],
         [-0.0087, -0.0012,  0.0021,  0.0044, -0.0098,  0.0016,  0.0011,
          -0.0111],
         [-0.0110,  0.0024, -0.0149,  0.0140,  0.0042,  0.0112, -0.0030,
           0.0135],
         [ 0.0109, -0.0029,  0.0086, -0.0060,  0.0109, -0.0020,  0.0013,
           0.0029]],

        [[-0.0093, -0.0018,  0.0034,  0.0030, -0.0108, -0.0002,  0.0018,
          -0.0139],
         [ 0.0042, -0.0010,  0.0013, -0.0107,  0.0049, -0.0097,  0.0021,
          -0.0007],
         [-0.0092, -0.0013,  0.0028,  0.0042, -0.0113,  0.0012,  0.0013,
          -0.0128],
         [ 0.0119, -0.0023,  0.0079, -0.0047,  0.0113, -0.0003,  0.0005,
           0.0050],
         [-0.0080, -0.0014,  0.0026,  0.0035, -0.0092,  0.0009,  0.0013,
          -0.0112],
         [ 0.0041,  0.0074, -0.0043,  0.0114, -0.0142,  0.0118, -0.0067,
           0.0058],
         [-0.0107, -0.0013,  0.0027,  0.0046, -0.0131,  0.0010,  0.0015,
          -0.0146],
         [ 0.0107,  0.0084, -0.0061,  0.0124, -0.0058,  0.0154, -0.0085,
           0.0168],
         [ 0.0099, -0.0024,  0.0079, -0.0047,  0.0091, -0.0010,  0.0009,
           0.0026],
         [ 0.0090, -0.0028,  0.0083, -0.0059,  0.0083, -0.0025,  0.0015,
           0.0006]]])

py_dK = torch.tensor([[[-1.4534e-03,  2.2734e-03,  1.0909e-03,  1.9949e-03,  8.2916e-04,
          -8.4295e-04],
         [-1.0866e-03, -6.1909e-03, -4.6506e-03, -6.8202e-03, -5.1510e-03,
          -4.6862e-03],
         [-8.7861e-04,  5.9759e-06, -6.3610e-04, -1.0830e-04, -1.2313e-03,
          -1.3148e-03],
         [ 5.8432e-03,  5.8214e-03,  6.3610e-03,  7.2490e-03,  9.2773e-03,
           1.0583e-02],
         [-1.6348e-03, -4.3599e-04, -1.6647e-03, -7.5878e-04, -2.6391e-03,
          -2.9721e-03],
         [ 1.2696e-04, -5.0342e-04,  3.3689e-04, -4.4700e-04,  1.7023e-04,
           6.8798e-04],
         [-7.8968e-04, -7.7373e-04, -7.6541e-04, -8.6670e-04, -1.2649e-03,
          -1.2298e-03],
         [ 5.6326e-05, -7.8531e-04, -1.5537e-04, -6.5867e-04, -1.8892e-04,
           1.4773e-04],
         [ 7.4304e-05,  1.1769e-04, -2.8401e-04, -6.9224e-05, -7.1934e-05,
          -4.4740e-04],
         [-2.5768e-04,  4.7096e-04,  3.6734e-04,  4.8500e-04,  2.7053e-04,
           7.4462e-05]],

        [[ 2.8167e-03,  1.0755e-03,  1.4125e-03,  1.1250e-03,  3.5121e-03,
           3.0757e-03],
         [-2.4070e-03, -2.0795e-03, -1.8462e-04, -1.7611e-03, -2.4637e-03,
          -9.0550e-04],
         [-4.8712e-04, -6.6046e-04,  5.9567e-04, -5.7974e-04, -7.4866e-05,
           6.2382e-04],
         [-1.0737e-03,  8.4999e-04, -2.6039e-04,  4.9693e-04, -1.2549e-03,
          -1.2456e-03],
         [-1.1289e-03, -1.2522e-03, -1.8273e-04, -1.4325e-03, -1.0399e-03,
          -7.5251e-04],
         [ 3.3067e-03,  3.4238e-03, -8.1084e-04,  3.2837e-03,  2.4513e-03,
           1.4110e-04],
         [-9.0135e-05, -7.4402e-05, -1.7517e-05, -6.3430e-05, -7.9707e-05,
          -5.7037e-05],
         [-6.5412e-05,  4.5692e-04,  2.1451e-04,  5.3120e-04,  1.0025e-04,
           1.5569e-04],
         [-2.4109e-04, -2.9751e-04, -1.6112e-04, -2.5625e-04, -2.7216e-04,
          -2.6875e-04],
         [-6.3007e-04, -1.4421e-03, -6.0545e-04, -1.3438e-03, -8.7837e-04,
          -7.6694e-04]]])


py_dV = torch.tensor([[[ 6.2295e-03, -2.7127e-03,  4.9296e-03, -1.4168e-02,  7.8000e-03,
          -1.2388e-02,  3.4841e-03, -2.6525e-03],
         [ 3.6854e-04,  9.4440e-03, -1.5674e-02,  2.1576e-02, -5.8116e-03,
           2.1683e-02, -9.7126e-03,  2.1809e-02],
         [-6.0407e-03,  2.4432e-03, -1.1235e-02,  1.0298e-02,  3.5056e-03,
           8.8978e-03, -2.8567e-03,  1.1977e-02],
         [-7.4424e-03,  8.7130e-03, -2.3998e-02,  2.5450e-02,  1.6242e-03,
           2.3738e-02, -9.3245e-03,  2.8637e-02],
         [-1.2084e-03,  1.9869e-04, -2.3145e-03,  1.7089e-03,  1.9314e-03,
           1.5041e-03, -3.6492e-04,  2.8690e-03],
         [-2.9282e-03,  3.0803e-04, -3.9102e-03,  2.2361e-03,  1.9447e-03,
           1.4897e-03, -3.2695e-04,  3.2439e-03],
         [-3.1632e-03,  3.9878e-04, -3.2138e-03,  3.8368e-03,  4.5104e-04,
           3.0423e-03, -6.3750e-04,  2.4847e-03],
         [-4.1402e-03, -2.2508e-04, -7.4724e-04,  3.0340e-03, -2.8375e-03,
           1.7818e-03,  8.7590e-05, -2.4088e-03],
         [-2.0161e-04, -8.5816e-06, -8.4830e-04,  9.8074e-04,  1.4840e-03,
           1.0602e-03, -2.0815e-04,  1.7667e-03],
         [ 1.6865e-03, -4.4482e-04,  1.3290e-03, -9.3704e-04,  1.6894e-03,
          -3.0586e-04,  1.9850e-04,  4.4705e-04]],

        [[-8.7936e-03, -3.7710e-03,  8.2588e-03, -8.3943e-04, -1.1361e-02,
          -3.8424e-03,  3.6979e-03, -1.9916e-02],
         [-5.3430e-05, -1.9477e-03,  4.4712e-03, -4.8502e-03, -1.0269e-03,
          -5.0044e-03,  2.1391e-03, -6.8299e-03],
         [-2.8873e-04, -1.0748e-03,  3.1053e-03,  2.3286e-04, -1.2665e-03,
           2.3812e-04,  6.3863e-04, -3.5084e-03],
         [ 3.5917e-03, -1.1099e-03,  3.7079e-03, -1.3999e-03,  2.9096e-03,
          -2.4690e-05,  4.1293e-04,  1.1642e-04],
         [-7.7682e-03, -1.3695e-03,  2.7420e-03,  3.5503e-03, -9.1488e-03,
           9.8488e-04,  1.2431e-03, -1.1057e-02],
         [ 1.1982e-02,  3.9971e-03,  5.5755e-03,  6.3663e-03, -7.5991e-03,
           9.8447e-03, -4.8492e-03,  3.5714e-03],
         [-7.2241e-04, -4.1172e-04,  1.1015e-03,  2.7101e-04, -1.1528e-03,
           5.1861e-05,  2.9900e-04, -1.9615e-03],
         [ 1.0710e-02,  8.3429e-03, -6.0394e-03,  1.2291e-02, -5.7444e-03,
           1.5280e-02, -8.4470e-03,  1.6723e-02],
         [ 1.4915e-03, -3.7181e-04,  1.2086e-03, -7.3939e-04,  1.3704e-03,
          -1.8379e-04,  1.4817e-04,  3.4923e-04],
         [ 2.5393e-03, -7.8045e-04,  2.3369e-03, -1.6562e-03,  2.3557e-03,
          -7.1782e-04,  4.1725e-04,  1.6690e-04]]])

py_dQ = torch.tensor([[[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
           0.0000e+00],
         [ 2.7272e-04,  4.1523e-04,  6.5173e-04,  3.6010e-04,  6.7076e-04,
           3.5940e-04],
         [ 1.3023e-03, -6.7120e-04, -9.6966e-05,  1.2825e-03, -5.2774e-04,
           3.9387e-04],
         [ 3.0620e-03, -8.4607e-04,  1.3031e-03,  3.3306e-03,  9.2511e-04,
           2.1579e-03],
         [ 2.7899e-03, -5.1652e-04,  1.7338e-03,  3.6767e-03,  1.2091e-03,
           2.4450e-03],
         [-3.6426e-04, -6.5674e-04, -3.4040e-04, -3.5949e-04, -4.0917e-04,
          -5.9803e-04],
         [ 1.9607e-03, -6.2041e-04,  1.1658e-03,  2.6651e-03,  8.6380e-04,
           1.8674e-03],
         [-8.9030e-04,  4.1701e-04, -5.9837e-04, -1.3690e-03, -2.7309e-04,
          -6.8057e-04],
         [ 5.6935e-03,  2.7823e-03,  5.1945e-03,  6.9949e-03,  3.8641e-03,
           4.6544e-03],
         [ 3.8944e-03,  2.5317e-03,  5.3678e-03,  4.2664e-03,  5.3066e-03,
           3.5115e-03]],

        [[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
           0.0000e+00],
         [-2.2735e-03, -1.1227e-03,  1.0984e-03, -1.3443e-03,  2.1672e-03,
          -5.1842e-05],
         [-5.4083e-04, -2.8037e-04,  2.4872e-04, -3.1000e-04,  4.8939e-04,
          -2.2103e-05],
         [ 2.2441e-03,  9.0294e-04,  6.8350e-04,  3.0991e-03, -1.2519e-03,
           3.3489e-04],
         [-1.5252e-04, -4.6346e-04, -6.7651e-04, -4.9649e-04, -4.8006e-04,
          -2.3406e-04],
         [ 1.9102e-06,  3.8510e-06,  4.7950e-07, -4.1904e-06,  4.9304e-06,
           2.2719e-06],
         [-1.1533e-02, -7.9764e-03, -1.0638e-02, -1.3551e-02, -5.5917e-03,
          -5.4541e-03],
         [ 1.4190e-04, -2.2598e-05,  1.9927e-04,  2.0728e-04,  2.3569e-04,
           2.2140e-04],
         [ 6.9615e-03,  4.7285e-03,  7.0335e-03,  8.1008e-03,  4.4664e-03,
           3.7497e-03],
         [ 4.5760e-05,  5.0518e-04,  1.4647e-03,  6.2870e-04,  1.4190e-03,
           4.5312e-04]]])

py_dX = torch.tensor([[[ 0.0094, -0.0064, -0.0041, -0.0015, -0.0157],
         [-0.0129,  0.0184,  0.0036,  0.0056,  0.0379],
         [ 0.0049,  0.0061,  0.0120,  0.0074,  0.0155],
         [ 0.0355,  0.0395,  0.0640,  0.0505,  0.0688],
         [ 0.0041,  0.0057,  0.0015,  0.0027,  0.0026],
         [ 0.0007, -0.0017,  0.0023,  0.0010,  0.0015],
         [ 0.0024,  0.0045,  0.0032,  0.0034,  0.0040],
         [-0.0064, -0.0044, -0.0034, -0.0046, -0.0029],
         [ 0.0185,  0.0180,  0.0127,  0.0173,  0.0123],
         [ 0.0172,  0.0163,  0.0100,  0.0155,  0.0089]],

        [[-0.0112, -0.0103, -0.0077, -0.0095, -0.0181],
         [-0.0080, -0.0095, -0.0155, -0.0107, -0.0172],
         [-0.0023, -0.0013, -0.0020, -0.0015, -0.0035],
         [ 0.0066,  0.0044,  0.0044,  0.0036,  0.0030],
         [-0.0169, -0.0103, -0.0119, -0.0135, -0.0138],
         [ 0.0014,  0.0226,  0.0144,  0.0119,  0.0296],
         [-0.0340, -0.0300, -0.0214, -0.0306, -0.0231],
         [ 0.0015,  0.0252,  0.0176,  0.0162,  0.0409],
         [ 0.0215,  0.0197,  0.0123,  0.0191,  0.0130],
         [ 0.0029,  0.0023, -0.0017,  0.0018, -0.0006]]])

"""
────────────────────────────────────────────────────────────────────────────────────
# Instantiate Attention object
────────────────────────────────────────────────────────────────────────────────────
"""

try:
  usr_attn = Attention(W_key, W_query, W_value)
except Exception as exc:
  print (traceback.format_exc())
  print (exc)

"""
────────────────────────────────────────────────────────────────────────────────────
# Forward Pass
────────────────────────────────────────────────────────────────────────────────────
"""

try:
  usr_out = usr_attn.forward(input)

  print("──────────────────────────────────────────")
  print("Attention outputs | STUDENT OUTPUT")
  print("──────────────────────────────────────────")

  print("\nAttention Output =\n", usr_out, sep="")
  print("\nK =\n", usr_attn.K, sep="")
  print("\nQ =\n", usr_attn.Q, sep="")
  print("\nV =\n", usr_attn.V, sep="")


  print("\n──────────────────────────────────────────")
  print("Attention outputs | SOLUTION OUTPUT")
  print("──────────────────────────────────────────")

  print("\nAttention output =\n", att_out, sep="")
  print("\nK =\n", K, sep="")
  print("\nQ =\n", Q, sep="")
  print("\nV =\n", V, sep="")


  print("\n──────────────────────────────────────────")
  print("Forward Pass | TEST RESULTS")
  print("──────────────────────────────────────────")

  print("\n           Pass?")

  TEST_attn_out = torch.allclose(usr_out, att_out, atol=1e-3)
  print("Test attention output:   ", TEST_attn_out)

  TEST_K = torch.allclose(K, usr_attn.K, atol=1e-3)
  print("Test K:   ", TEST_K)

  TEST_Q = torch.allclose(Q, usr_attn.Q, atol=1e-3)
  print("Test Q:   ", TEST_Q)

  TEST_V = torch.allclose(V, usr_attn.V, atol=1e-3)
  print("Test V:   ", TEST_V)

except Exception as exc:
  print (traceback.format_exc())
  print (exc)


"""
────────────────────────────────────────────────────────────────────────────────────
# Backward Pass
────────────────────────────────────────────────────────────────────────────────────
"""

try:
  usr_dLdX = usr_attn.backward(dLXnew)

  print("──────────────────────────────────────────")
  print("Backward Pass | STUDENT OUTPUT")
  print("──────────────────────────────────────────")

  print("\ndLdK =\n", usr_attn.dLdK, sep="")
  print("\ndLdQ =\n", usr_attn.dLdQ, sep="")
  print("\ndLdV =\n", usr_attn.dLdV, sep="")
  print("\ndLdX =\n", usr_dLdX, sep="")


  print("\n──────────────────────────────────────────")
  print("Backward Pass | SOLUTION OUTPUT")
  print("──────────────────────────────────────────")

  print("\ndLdK =\n", py_dK, sep="")
  print("\ndLdQ =\n", py_dQ, sep="")
  print("\ndLdV =\n", py_dV, sep="")
  print("\ndLdX =\n", py_dX, sep="")
  
  print("\n──────────────────────────────────────────")
  print("Backward Pass | TEST RESULTS")
  print("──────────────────────────────────────────")

  print("           Pass?")

  TEST_dK = torch.allclose(usr_attn.dLdK, py_dK, atol=1e-3)
  print("Test dLdK:   ", TEST_dK)
  TEST_dQ = torch.allclose(usr_attn.dLdQ, py_dQ, atol=1e-3)
  print("Test dLdQ:   ", TEST_dQ)
  TEST_dV = torch.allclose(usr_attn.dLdV, py_dV, atol=1e-3)
  print("Test dLdV:   ", TEST_dV)
  TEST_dX = torch.allclose(usr_dLdX, py_dX, atol=1e-3)
  print("Test dLdX:   ", TEST_dX)

except Exception as exc:
  print (traceback.format_exc())
  print (exc)


"""
────────────────────────────────────────────────────────────────────────────────────
## SCORE AND GRADE TESTS
────────────────────────────────────────────────────────────────────────────────────
"""

TEST_forward = (
    TEST_attn_out and
    TEST_Q and
    TEST_K and
    TEST_V
)

TEST_backward = (
    TEST_dX and
    TEST_dV and
    TEST_dQ and
    TEST_dK)


SCORE_LOGS = {
    "Forward": 10 * int(TEST_forward),
    "Backward": 10 * int(TEST_backward)
}


print("\n")
print("TEST   | STATUS | POINTS | DESCRIPTION")
print("───────┼────────┼────────┼────────────────────────────────")

for i, (key, value) in enumerate(SCORE_LOGS.items()):

    index_str = str(i).zfill(1)
    point_str = str(value).zfill(2) + "     │ "

    if value == 0:
        status_str = " │ FAILED │ "
    else:
        status_str = " │ PASSED │ "

    print("Test ", index_str, status_str, point_str, key, sep="")

print("\n")

"""
────────────────────────────────────────────────────────────────────────────────────
## FINAL SCORES
────────────────────────────────────────────────────────────────────────────────────
"""

print(json.dumps({'scores': SCORE_LOGS}))