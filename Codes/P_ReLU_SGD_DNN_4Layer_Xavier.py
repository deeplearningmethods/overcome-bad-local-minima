from matplotlib.ticker import PercentFormatter
from matplotlib.gridspec import GridSpec
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

train_steps = 10000; T = train_steps; batch_size = 1024; l_rate = 0.01
Initializations = 300; M = Initializations; Dim = [1, 50, 50, 1]

# Realization function associated to deep ANNs
var_list, d, Rn, Var = [], [], tf.random.normal, tf.Variable
def realization(x, activation, nns):
    def layer(y, h0, h1, act):
# Initialize ANN parameters: Xavier normal
        W = Var(Rn((M, h0, h1), stddev=tf.math.sqrt(2/(h0+h1))), 'weights')
        B = Var(tf.zeros((M, 1, h1)), 'biases')
        var_list.append(W); d.append(h0*h1)
        var_list.append(B); d.append(h1)
        return act(tf.linalg.matmul(x, W)+B)
    for i in range(len(nns)-2):
        x = layer(x, nns[i], nns[i+1], activation)
    return layer(x, nns[len(nns)-2], nns[len(nns)-1], tf.identity)

# Activation; GD Optimizer; Target function; Risk function
ReLU = tf.nn.relu; Opt = tf.compat.v1.train.GradientDescentOptimizer
def f(x):
    return x**2 + 2*x

X = tf.random.uniform((batch_size, Dim[0]), minval=0, maxval=1, seed=5)
L = tf.reduce_mean((realization(X, ReLU, Dim) - f(X))**2, 1)
Opt1 = Opt(l_rate).minimize(L)

# ANN parameter; Gradient
par = var_list; g = tf.gradients(L, par)
for i in range(len(var_list)):
    par[i] = tf.reshape(par[i], [M,d[i]])
    g[i] = tf.reshape(g[i], [M,d[i]])
parr, gg = tf.concat(par, 1), tf.concat(g, 1)
grad = tf.norm(gg, 'euclidean', [1,0])/(M**(1/2))

with tf.compat.v1.Session() as sess:

    sess.run(tf.compat.v1.global_variables_initializer())
    Grad, p, P = [], [], []
    for i in range(T):
        sess.run(Opt1)
        if (i%10==0):
            p.append(parr.eval()); Grad.append(grad.eval())
    for i in range(len(p)-1):
        P.append(tf.norm(p[i]-p[-1], 'euclidean', [1,0])/(M**(1/2)))
    Param = sess.run(P); graph = sess.run(L)

#----------------------- Numerical Simulation -----------------------
font1 = {'fontname':'Perpetua', 'size':16}
font2 = {'fontname':'Perpetua', 'size':11}; splt = plt.subplot
gs = GridSpec(2, 2, width_ratios=[2, 0.75], height_ratios=[1, 1])
gs.update(top=1, right=1.8, wspace=0.17, hspace=0.35)
fig1, fig2, fig3 = splt(gs[:, 0]), splt(gs[0, 1]), splt(gs[1, 1])
C, B = np.histogram(graph, bins=80)
fig1.hist(B[:-1], B, weights=C/M, color="teal", edgecolor="darkgreen")
fig1.yaxis.set_major_formatter(PercentFormatter(1))
fig1.set_xlabel('Risk levels', fontdict=font1)
fig1.set_ylabel('Relative frequency of risk levels', fontdict=font1)
xx=np.linspace(1, T, len(Param), dtype=int)
fig2.loglog(xx, Grad[0:len(Grad)-1], color='teal')
gm=int(np.log10(np.min(Grad))); gM=int(np.log10(np.max(Grad)))
fig2.set_yticks(np.logspace(gm, gM, num=gM-gm+1))
fig2.set_xlabel('Number of SGD training steps', fontdict=font2)
fig2.set_ylabel('Norm of the gradient', fontdict=font2)
fig3.semilogx(xx, Param, color = 'teal')
fig3.set_xlabel('Number of SGD training steps', fontdict=font2)
fig3.set_ylabel(r'$\mathcal{L}^{2}\!$'+'-distance to limit point', fontdict=font2)
plt.show
plt.savefig("ReLU_DNN_SGD_1_50_50_1_Xavier.pdf", bbox_inches="tight")