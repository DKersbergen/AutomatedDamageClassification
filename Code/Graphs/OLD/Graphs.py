import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import seaborn as sns
"""
np.random.seed(1974)

# Generate Data
num = 20
x, y = np.random.random((2, num))
labels = np.random.choice(['a', 'b', 'c'], num)
df = pd.DataFrame(dict(x=x, y=y, label=labels))

groups = df.groupby('label')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
ax.legend()

plt.show()


location = 'HistMatch/Damage_Build_Info.csv'
df = pd.read_csv(location)

groups = df.groupby('_damage')

fig = plt.figure()
num = 1
for key in df.keys()[7:]:
    if key[-4:] == 'mean':
        for name, group in groups:
            print(name)
            print(key)
            ax = fig.add_subplot(4,3,num)
            ax.plot(group., y = key[:-4]+"Range", marker='o', linestyle='', ms=12, label=name)
            #group.plot.scatter(x = key, y = key[:-4]+"Range", marker='o', label=name)
        ax.legend()
        num += 1
#plt.tight_layout()
plt.show()



"""
location = 'HistMatch/Damage_Build_Info.csv'
df = pd.read_csv(location)
keyList = df.keys().tolist()
print(keyList)
"""
print(keyList[5:7])
keyListShort = []
keyListShort = [keyList[5]] + keyList[8:14]
keyListShort2 = [keyList[5]] + keyList[15:]
print(keyListShort)
print(keyListShort2)
sns.pairplot(df[keyListShort], hue = "_damage")
sns.pairplot(df[keyListShort2], hue = "_damage")

my_dpi = 96
fig = plt.figure(figsize=(800/my_dpi,800/my_dpi), dpi = my_dpi)
num = 1
for key in keyList[7:]:
    if key[-6:] == 'median' or key[-4:] == 'mean' or key[-5:] == 'stdev':
        ax = fig.add_subplot(6,5,num)
        df.plot.scatter(x=key, y='damage_num',ax = ax, legend=False)
        df.set_index(key, inplace=True)
        plt.title(key)
        plt.axis([0,1,-1,4])
        num += 1

fig.tight_layout()
plt.savefig('Test.png', dpi = my_dpi*10)
plt.show()

with open('HistMatch/Damage_Build_Info.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    dictKeys = next(reader).keys()
    for dictKey in dictKeys:
        Damage.append(float(row['DamageNo']))
        Value.append(float(row['Value']))

y = np.array(Damage)
print(y)
x = np.array(Value)
print(x)
z = np.polyfit(x,y,3)
print(z)

f = np.poly1d(z)
print(f)
x_new = np.linspace(0, 2, 50)
y_new = f(x_new)
ax = plt.gca()
ax.set_axis_bgcolor((0.898, 0.898, 0.898))
plt.gcf()
plt.plot(x,y, 'ro', x_new, y_new)
plt.show()
"""