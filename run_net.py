from __future__ import division, print_function
import time
import torch
from src.utils import *
from src.models import Net

# --------------------
# instantiate net
input_dim = 2
output_dim = 2
n_hid = 2
use_cuda = torch.cuda.is_available()
lr = 0.001
net = Net(input_dim, output_dim, n_hid, lr, use_cuda)

## ---------------------------------------------------------------------------------------------------------------------
# train
cprint('c','\nTrain:')


x_train = np.ones((1000,2))
x_dev = np.ones((100,2))

y_train = np.ones((1000,2))
y_dev = np.ones((100,2))

batch_size = 1
nb_epochs = 2

cost_train = np.zeros(nb_epochs)
cost_dev = np.zeros(nb_epochs)

nb_samples_train = len(x_train)
nb_samples_dev = len(x_dev)

best_cost = np.inf
nb_its_dev = 1
tic0 = time.time()
for i in range(nb_epochs):
    net.set_mode_train(True)
    # ---- W
    tic = time.time()
    #for x, y in trainloader:
    for ind in generate_ind_batch(nb_samples_train, batch_size):
        x, y = x_train[ind], y_train[ind]
        loss  = net.fit(x, y)
        cost_train[i] += loss / nb_samples_train * len(x)

    toc = time.time()

    # ---- print
    print("it %d/%d, Jtr = %f " % (i, nb_epochs, cost_train[i]), end="")
    cprint('r','   time: %f seconds\n' % (toc - tic))
    net.update_lr(i)

    # ---- dev
    if i % nb_its_dev == 0:
        net.set_mode_train(False)
        #print('eval mode on')
        #for x, y in testloader:
        for ind in generate_ind_batch(nb_samples_dev, batch_size, random=False):
            x, y = x_dev[ind], y_dev[ind]
            cost =  net.eval(x, y)
            cost_dev[i] += cost / nb_samples_dev * len(x)
        cprint('g','    Jdev = %f\n' % (cost_dev[i]))
        if cost_dev[i] < best_cost:
            best_cost = cost_dev[i]
            net.save('./theta_best.dat')

toc0 = time.time()
runtime_per_it =  (toc0 - tic0)/float(nb_epochs)
cprint('r','   average time: %f seconds\n' % runtime_per_it)

## ---------------------------------------------------------------------------------------------------------------------
# save model
net.save('./theta_last.dat')

## ---------------------------------------------------------------------------------------------------------------------
# results
cprint('c','\nRESULTS:')
cost_dev_min = cost_dev[::nb_its_dev].min()
cost_train_min = cost_train.min()
nb_parameters = net.get_nb_parameters()
print('  cost_dev: %f (cost_train %f)' % (cost_dev_min, cost_train_min))
print('  nb_parameters: %d (%s)' % (nb_parameters, humansize(nb_parameters)))
print('  time_per_it: %fs\n' % (runtime_per_it))