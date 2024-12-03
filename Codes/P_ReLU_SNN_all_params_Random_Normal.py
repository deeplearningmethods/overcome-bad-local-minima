from matplotlib.ticker import PercentFormatter
from matplotlib.gridspec import GridSpec
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution ()

train_steps = 10000; T = train_steps; batch_size = 1024
Dim1 = [1, 10, 1]; Dim2 = [1, 100, 1]; Dim3 = [1, 1000, 1]
Initializations = 300; M = Initializations; k = 2*(len(Dim1) - 1)

# Realization function associated to shallow ANNs
var_list, d, Rn, Var = [], [], tf.random.normal, tf.Variable
def realization(x, activation, nns):
    def layer(y, h0, h1, act):
# Initialize ANN parameters: standard normal
        W = Var(Rn((M, h0, h1)), 'weights')
        B = Var(Rn((M, 1, h1)), 'biases')
        var_list.append(W); d.append(h0*h1)
        var_list.append(B); d.append(h1)
        return act(tf.linalg.matmul(y, W)+B)
    for i in range(len(nns)-2):
        x = layer(x, nns[i], nns[i+1], activation)
    return layer(x, nns[len(nns)-2], nns[len(nns)-1], tf.identity)


# Activation; GD Optimizer; Target function; Risk function
ReLU = tf.nn.relu; Opt = tf.compat.v1.train.GradientDescentOptimizer
def f(x):
    return tf.sin(3*x)

X = tf.random.uniform((batch_size, 1), minval =0, maxval =1, seed =5)
L1 = tf.reduce_mean((realization(X, ReLU, Dim1) - f(X))**2, 1)
L2 = tf.reduce_mean((realization(X, ReLU, Dim2) - f(X))**2, 1)
L3 = tf.reduce_mean((realization(X, ReLU, Dim3) - f(X))**2, 1)

learning_rate = tf.compat.v1.placeholder(tf.float32, shape=())
Opt1 = Opt(learning_rate=learning_rate).minimize(L1)
Opt2 = Opt(learning_rate=learning_rate).minimize(L2)
Opt3 = Opt(learning_rate=learning_rate).minimize(L3)

# ANN parameter; Gradient
par1 = var_list[0:k]; g1 = tf.gradients(L1, par1)
par2 = var_list[k:2*k]; g2 = tf.gradients(L2, par2)
par3 = var_list[2*k:3*k]; g3 = tf.gradients(L3, par3)

for i in range(k):
    par1[i] = tf.reshape(par1[i], [M,d[i]])
    par2[i] = tf.reshape(par2[i], [M,d[k+i]])
    par3[i] = tf.reshape(par3[i], [M,d[2*k+i]])
    g1[i] = tf.reshape(g1[i], [M,d[i]])
    g2[i] = tf.reshape(g2[i], [M,d[k+i]])
    g3[i] = tf.reshape(g3[i], [M,d[2*k+i]])
parr1, gg1 = tf.concat(par1, 1), tf.concat(g1, 1)
parr2, gg2 = tf.concat(par2, 1), tf.concat(g2, 1)
parr3, gg3 = tf.concat(par3, 1), tf.concat(g3, 1)

grad1 = tf.norm(gg1, 'euclidean', [1,0])/(M**(1/2))
grad2 = tf.norm(gg2, 'euclidean', [1,0])/(M**(1/2))
grad3 = tf.norm(gg3, 'euclidean', [1,0])/(M**(1/2))

with tf.compat.v1.Session() as sess:

    sess.run(tf.compat.v1.global_variables_initializer())
    Grad1, p1, P1 = [], [], []
    Grad2, p2, P2 = [], [], []
    Grad3, p3, P3 = [], [], []

    for i in range(T):
        l_rate =  0.001*(i+1)**(-1/10)
        sess.run(Opt1, feed_dict={learning_rate: l_rate})
        sess.run(Opt2, feed_dict={learning_rate: l_rate})
        sess.run(Opt3, feed_dict={learning_rate: l_rate})
        if (i%20==0):
            p1.append(parr1.eval()); Grad1.append(grad1.eval())
            p2.append(parr2.eval()); Grad2.append(grad2.eval())
            p3.append(parr3.eval()); Grad3.append(grad3.eval())

    for i in range(len(p1)-1):
        P1.append(tf.norm(p1[i]-p1[-1], 'euclidean', [1,0])/(M**(1/2)))
        P2.append(tf.norm(p2[i]-p2[-1], 'euclidean', [1,0])/(M**(1/2)))
        P3.append(tf.norm(p3[i]-p3[-1], 'euclidean', [1,0])/(M**(1/2)))
    Param1, Param2, Param3 = sess.run(P1), sess.run(P2), sess.run(P3)
    graph1, graph2, graph3 = sess.run(L1), sess.run(L2), sess.run(L3)

# ----------------------- Numerical Simulation -----------------------
font1 = {'fontname':'Perpetua', 'size':16}
font2 = {'fontname':'Perpetua', 'size':11}; splt = plt.subplot
gs = GridSpec(2, 2, width_ratios=[2, 0.75], height_ratios=[1, 1])
gs.update(top=1, right=1.8, wspace=0.17, hspace=0.35)
fig1, fig2, fig3 = splt(gs[:, 0]), splt(gs[0, 1]), splt(gs[1, 1])
C1, B1 = np.histogram(graph1, bins=40)
fig1.hist(B1[:-1], B1, weights=C1/M, color="gray", edgecolor="darkgray",
          label=r'$(\ell_0, \ell_1, \ell_2) =$'+format(tuple(Dim1)))
C2, B2 = np.histogram(graph2, bins=40)
fig1.hist(B2[:-1], B2, weights=C2/M, color="orange", edgecolor="darkorange",
          label=r'$(\ell_0, \ell_1, \ell_2) =$'+format(tuple(Dim2)))
C3, B3 = np.histogram(graph3, bins=40)
fig1.hist(B3[:-1], B3, weights=C3/M, color="teal", edgecolor="darkgreen",
          label=r'$(\ell_0, \ell_1, \ell_2) =$'+format(tuple(Dim3)))
fig1.yaxis.set_major_formatter(PercentFormatter(1))
fig1.set_xscale('log', base=10)
fig1.legend()
fig1.set_xlabel('Risk levels', fontdict=font1)
fig1.set_ylabel('Relative frequency of risk levels', fontdict=font1)
xx=np.linspace(1, T, len(Param1), dtype=int)
fig2.loglog(xx, Grad1[0:len(Grad1)-1], color='gray')
fig2.loglog(xx, Grad2[0:len(Grad2)-1], color='darkorange')
fig2.loglog(xx, Grad3[0:len(Grad3)-1], color='teal')
fig2.set_xlabel('Number of SGD training steps', fontdict=font2)
fig2.set_ylabel('Norm of the gradient', fontdict=font2)
fig3.semilogx(xx, Param1, color = 'gray')
fig3.semilogx(xx, Param2, color = 'darkorange')
fig3.semilogx(xx, Param3, color = 'teal')
fig3.set_xlabel('Number of SGD training steps', fontdict=font2)
fig3.set_ylabel(r'$\mathcal{L}^{2}\!$'+'-distance to limit point', fontdict=font2)
plt.show
plt.savefig("Opt_SNN_SGD_10_100_1000_ReLU_all_params_train.pdf", bbox_inches="tight")