import matplotlib.pyplot as plt
import matplotlib


fig = plt.figure()
ax = fig.gca()
ax.plot(train_loss_vector, linewidth=5)
plt.title('Train Loss')
#matplotlib.rc('Train Loss', size=14)

fig = plt.figure()
ax = fig.gca()
ax.plot(test_loss_vector, linewidth=5)
plt.title('Test Loss')
#matplotlib.rc('Test Loss', size=14)

fig = plt.figure()
ax = fig.gca()
ax.plot(test_accuracy, linewidth=5)
plt.title('Test Accuracy (%)')
#matplotlib.rc('Test Accuracy', size=14)


