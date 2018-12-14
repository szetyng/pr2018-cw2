import matplotlib.pyplot as plt

y_eucl = [47.0, 66.86, 74.93, 79.07, 83.21]
y_cosine = [47.57,67,75.07,79.43,82.71]
y_corr = [46.93,66.36,74.29,78.29,82.21]
y_manhattan = [47.21, 66.14, 75.07, 79.79, 82.64]
x = [1,5,10,15,20]

plt.scatter(x, y_eucl)
plt.plot(x, y_eucl, label='euclidean')
plt.scatter(x, y_cosine)
plt.plot(x, y_cosine, label='cosine')
plt.scatter(x, y_corr)
plt.plot(x, y_corr, label='correlation')
plt.scatter(x, y_manhattan)
plt.plot(x, y_manhattan, label='manhattan')

plt.xlabel('k')
plt.ylabel('Accuracy / %')
plt.xlim(1,20)
plt.ylim(45,85)
plt.legend(loc = 'right')
plt.show()