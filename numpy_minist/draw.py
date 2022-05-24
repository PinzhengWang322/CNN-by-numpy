import matplotlib.pyplot as plt
import pylab
cnn_x, cnn_y, l_x, l_y = [], [], [], [] 
with open("cnn_record", "r") as f:
    for idx, data in enumerate(f.readlines()):
        if idx % 2 == 0: continue
        cnn_x.append((idx * 10 + 10) / 459)
        cnn_y.append(float(data))

with open("linear_record", "r") as f2:
    for idx, data in enumerate(f2.readlines()):
        if idx % 2 == 0: continue
        l_x.append((idx * 10 + 10) / 459)
        l_y.append(float(data))

plt.plot(cnn_x, cnn_y,  linewidth=2, label="CNN_model")
plt.plot(l_x, l_y,  linewidth=2, label="Linear_model")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.tick_params(labelsize=8)
plt.tight_layout()
plt.legend(bbox_to_anchor=(0.28, 1.1), loc=2, borderaxespad=0,prop = {'size':10},ncol=2)
plt.tick_params(labelsize=8)
plt.tight_layout()
plt.grid(c = 'linen')


plt.show()
pylab.show()