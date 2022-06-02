# importing relevant libraries
import matplotlib.pyplot as plt
from scipy import stats

# data for training
x = [240,265,235,495,310,250,510,310,290,175]
y = [412,501,520,660,445,377,735,375,567,356]

# calling LRM function for relevant objects
slope, intercept, r, p, std_err = stats.linregress(x, y)

# y = mx + C
def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

# input value for value prediction
speed = myfunc(500)

# printing relevant data
print(speed)
print(intercept)

# plot graph
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()