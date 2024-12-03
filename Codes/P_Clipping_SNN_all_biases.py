from matplotlib.ticker import PercentFormatter
from matplotlib.gridspec import GridSpec
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution ()

train_steps = 10000; T = train_steps; batch_size = 1024
Initializations = 300; M = Initializations; h1 = 10; h2 = 100; h3 = 1000

# Realization function associated to shallow ANNs
var_list, d, Rn, Var = [], [], tf.random.normal, tf.Variable
def realization(x, activation, h):
# Initialize ANN parameters: standard normal like
    w = Var((h**(3))*abs(Rn((M, 1, h))), name='inner_weights', trainable=False)
    b = Var((h**(3))*Rn((M, 1, h)), name='inner_biases', trainable=True)
    v = Var((h**(-7/8))*abs(Rn((M, h, 1))), name='outer_weights', trainable=False)
    c = Var(Rn((M, 1, 1)), name='outer_bias', trainable=True)
    var_list.append(b); d.append(h)
    var_list.append(c); d.append(1); mult = tf.linalg.matmul
    return (c + mult(activation(b + mult(x, w)), v))

# Activation; GD Optimizer; Target function; Risk function
def clipping(x):
    return tf.math.minimum(tf.math.maximum(x, 0), 1)
Opt = tf.compat.v1.train.GradientDescentOptimizer
def f(x):
    return x**(1/4)

X = tf.random.uniform((batch_size, 1), minval=0, maxval=1, seed=5)
L1 = tf.reduce_mean((realization(X, clipping, h1) - f(X))**2, 1)
L2 = tf.reduce_mean((realization(X, clipping, h2) - f(X))**2, 1)
L3 = tf.reduce_mean((realization(X, clipping, h3) - f(X))**2, 1)

learning_rate = tf.compat.v1.placeholder(tf.float32, shape=())
Opt1 = Opt(learning_rate=learning_rate).minimize(L1)
Opt2 = Opt(learning_rate=learning_rate).minimize(L2)
Opt3 = Opt(learning_rate=learning_rate).minimize(L3)

# ANN parameter; Gradient
par1 = var_list[0:2]; g1 = tf.gradients(L1, par1)
par2 = var_list[2:4]; g2 = tf.gradients(L2, par2)
par3 = var_list[4:6]; g3 = tf.gradients(L3, par3)

for i in range(2):
    par1[i] = tf.reshape(par1[i], [M,d[i]])
    par2[i] = tf.reshape(par2[i], [M,d[2+i]])
    par3[i] = tf.reshape(par3[i], [M,d[4+i]])
    g1[i] = tf.reshape(g1[i], [M,d[i]])
    g2[i] = tf.reshape(g2[i], [M,d[2+i]])
    g3[i] = tf.reshape(g3[i], [M,d[4+i]])
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
        l_rate = 0.01*0.5**(int((i+1)/150))
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

#----------------------- Numerical Simulation -----------------------
font1 = {'fontname':'Perpetua', 'size':16}
font2 = {'fontname':'Perpetua', 'size':11}; splt = plt.subplot
gs = GridSpec(2, 2, width_ratios=[2, 0.75], height_ratios=[1, 1])
gs.update(top=1, right=1.8, wspace=0.17, hspace=0.35)
fig1, fig2, fig3 = splt(gs[:, 0]), splt(gs[0, 1]), splt(gs[1, 1])
C1, B1 = np.histogram(graph1, bins=40)
fig1.hist(B1[:-1], B1, weights=C1/M, color="gray", edgecolor="darkgray",
          label=r'$(\ell_0, \ell_1, \ell_2) =$'+format(tuple([1,h1,1])))
C2, B2 = np.histogram(graph2, bins=40)
fig1.hist(B2[:-1], B2, weights=C2/M, color="orange", edgecolor="darkorange",
          label=r'$(\ell_0, \ell_1, \ell_2) =$'+format(tuple([1,h2,1])))
C3, B3 = np.histogram(graph3, bins=40)
fig1.hist(B3[:-1], B3, weights=C3/M, color="teal", edgecolor="darkgreen",
          label=r'$(\ell_0, \ell_1, \ell_2) =$'+format(tuple([1,h3,1])))
fig1.yaxis.set_major_formatter(PercentFormatter(1))
fig1.set_xscale('log', base=10)
fig1.legend()
fig1.set_xlabel('Risk levels', fontdict=font1)
fig1.set_ylabel('Relative frequency of risk levels', fontdict=font1)
xx=np.linspace(1, T, len(Param1), dtype=int)
fig2.loglog(xx, Grad1[0:len(Grad1)-1], color='gray')
fig2.loglog(xx, Grad2[0:len(Grad2)-1], color='darkorange')
fig2.loglog(xx, Grad3[0:len(Grad3)-1], color='teal')
gm=int(np.log10(np.min([Grad1, Grad2, Grad3])))
gM=int(np.log10(np.max([Grad1, Grad2, Grad3])))
fig2.set_yticks(np.logspace(gm, gM, num=gM-gm+1))
fig2.set_xlabel('Number of SGD training steps', fontdict=font2)
fig2.set_ylabel('Norm of the gradient', fontdict=font2)
fig3.semilogx(xx, Param1, color = 'gray')
fig3.semilogx(xx, Param2, color = 'darkorange')
fig3.semilogx(xx, Param3, color = 'teal')
fig3.set_xlabel('Number of SGD training steps', fontdict=font2)
fig3.set_ylabel(r'$\mathcal{L}^{2}\!$'+'-distance to limit point', fontdict=font2)
plt.show
plt.savefig("Opt_SNN_SGD_10_100_1000_clipping_biases_train.pdf", bbox_inches="tight")