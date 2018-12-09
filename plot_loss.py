import pickle
import matplotlib.pyplot as plt

with open('/home/udit/hvd-single-random/plot_loss/acc_1024/2p.pkl', 'rb') as f:
	track_loss_2 = pickle.load(f)
with open('/home/udit/hvd-single-random/plot_loss/acc_1024/2p-8sa.pkl', 'rb') as f:
	track_loss_4 = pickle.load(f)
# with open('/home/udit/hvd-single-random/plot_loss/acc_1024/scaled_8.pkl', 'rb') as f:
# 	track_loss_8 = pickle.load(f)
# with open('/home/udit/hvd-single-random/plot_loss/acc_1024/scaled_16.pkl', 'rb') as f:
# 	track_loss_16 = pickle.load(f)

fig = plt.figure()
ax = fig.gca()
#ax.set_xticks(numpy.arange(0, 1, 0.1))
#plt.yticks(np.arange(min(x), max(x)+1, 1.0))

plt.plot(*zip(*track_loss_2), label = 'Default behaviour')
plt.plot(*zip(*track_loss_4), label = 'Delayed Synchronisation')
plt.title('Delayed Synchronisation after 8 steps across 2 workers')
plt.xlabel('Number of Iterations')
plt.ylabel('Training Loss')
plt.legend()
plt.grid()
plt.show()