import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
# np.random.seed(19680801)


# N = 50
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
#
N = 10


# x1 = np.full((10),0)
# y1 = [109.0, 100.0, 67.0, 93.0, 69.0, 110.0, 89.0, 69.0, 87.0, 85.0]
# #area = (30 * np.random.rand(N))**2 
# plt.scatter(x1, y1, alpha=0.5)
# x1 = np.full((10), 10)
# y1 = [18.0, 14.0, 10.0, 16.0, 26.0, 14.0, 22.0, 38.0, 26.0, 16.0]
# #area = (30 * np.random.rand(N))**2 
# plt.scatter(x1, y1, alpha=0.5)
x1 = np.full((20),0)
y1 = [78.0, 112.0, 101.0, 83.0, 70.0, 75.0, 143.0, 119.0, 91.0, 73.0, 109.0, 100.0, 104.0, 69.0, 55.0, 61.0, 177.0, 86.0, 109.0, 60.0]
#area = (30 * np.random.rand(N))**2 
plt.scatter(x1, y1, color = 'green',alpha=0.5)

x1 = np.full((20),0)
y1 = [78.0, 112.0, 101.0, 83.0, 70.0, 75.0, 143.0, 119.0, 91.0, 73.0, 109.0, 100.0, 104.0, 69.0, 55.0, 61.0, 177.0, 86.0, 109.0, 60.0]
#area = (30 * np.random.rand(N))**2 
plt.scatter(x1, y1, color = 'red',alpha=0.5)

x1 = np.full((10), 20)
y1 = [43.0, 31.0, 14.0, 46.0, 22.0, 27.0, 17.0, 22.0, 27.0, 21.0]
#area = (30 * np.random.rand(N))**2 
plt.scatter(x1, y1, alpha=0.5, color = 'blue')

x1 = np.full((10), 25)
y1 = [42.0, 50.0, 29.0, 32.0, 70.0, 47.0, 32.0, 75.0, 50.0, 51.0]
#area = (30 * np.random.rand(N))**2 
plt.scatter(x1, y1, alpha=0.5, color = 'blue')

x1 = np.full((10), 30)
y1 = [66.0, 32.0, 52.0, 40.0, 137.0, 30.0, 64.0, 41.0, 57.0, 60.0]
#area = (30 * np.random.rand(N))**2 
plt.scatter(x1, y1, alpha=0.5, color = 'blue')

x1 = np.full((10), 35)
y1 = [51.0, 43.0, 38.0, 68.0, 37.0, 50.0, 34.0, 91.0, 82.0, 63.0]
#area = (30 * np.random.rand(N))**2 
plt.scatter(x1, y1, alpha=0.5, color = 'blue')

x1 = np.full((10), 40)
y1 = [61.0, 138.0, 47.0, 94.0, 122.0, 110.0, 60.0, 50.0, 89.0, 61.0]
#area = (30 * np.random.rand(N))**2 
plt.scatter(x1, y1, alpha=0.5, color = 'blue')

x1 = np.full((10), 45)
y1 = [121.0, 60.0, 152.0, 103.0, 55.0, 156.0, 128.0, 22.0, 46.0, 36.0]
#area = (30 * np.random.rand(N))**2 
plt.scatter(x1, y1, alpha=0.5, color = 'blue')

x1 = np.full((10), 50)
y1 = [43.0, 65.0, 50.0, 77.0, 72.0, 79.0, 47.0, 56.0, 23.0, 75.0]
#area = (30 * np.random.rand(N))**2 
plt.scatter(x1, y1, alpha=0.5, color = 'blue')

x1 = np.full((10), 55)
y1 = [37.0, 68.0, 71.0, 153.0, 58.0, 66.0, 52.0, 43.0, 56.0, 62.0]
#area = (30 * np.random.rand(N))**2 
plt.scatter(x1, y1, alpha=0.5, color = 'blue')


plt.savefig("./scatter1.png")