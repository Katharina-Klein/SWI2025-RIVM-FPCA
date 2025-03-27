import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from scipy.interpolate import BSpline
import skfda
from skfda.representation.interpolation import SplineInterpolation
from skfda.exploratory.visualization import FPCAPlot
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import (
    BSplineBasis,
)
from skfda.misc.metrics import MahalanobisDistance

from data import *


filename = "Data/COVID-19_rioolwaterdata.csv"

data_dict = data_to_dict(filename)

data = data_dict["Data"]
days = data_dict["Days"]
dates = data_dict["Dates"]
start_date = data_dict["Start date"]
locations = data_dict["Locations"]


loc = 'Holten'
index = locations.index(loc)

_, loc_data, loc_days, loc_dates = get_loc_data(data_dict, index)

print('Station: '+loc)


### Visualize data ###


show_min = loc_days.min()
show_max = loc_days.max()

plt.figure()
plt.plot(loc_data[0], loc_data[1])
plt.xticks(loc_days[::40], labels=loc_dates[::40], rotation=70)
plt.xlim([show_min, show_max])
plt.title('Measured viral load - '+loc)
plt.show()




### Define period to consider

from_date = '2022-01-30'

num_intervals = 50
period_length = 7




from_day = date_to_day_index(from_date, start_date)
to_day = from_day + num_intervals*period_length
to_date = day_index_to_date(to_day, start_date)

print('Start date of period: '+from_date)
print('End date of period: '+to_date)


num_points = 5*period_length


# Make plot of analysis period

show = [day for day in loc_days if from_day <= day <= to_day]
show_min = show[0]-period_length
show_max = show[-1]+period_length


plt.figure(figsize=(12,4))
plt.plot(loc_data[0], loc_data[1], '.-', linewidth=0.5)
for i in range(num_intervals):
    plt.axvline(from_day+(i*period_length), color='grey', linestyle='--', linewidth=0.5)
    plt.axvline(from_day+((i+1)*period_length), color='grey', linestyle='--', linewidth=0.5)
plt.xticks(loc_days[::20], labels=loc_dates[::20], rotation=70)
plt.xlim([show_min, show_max])
#plt.ylabel('')
plt.title('Measured viral load - '+loc+' - '+from_date+' until '+to_date)




### Create functions


fd = skfda.FDataGrid(grid_points=loc_data[0],data_matrix=loc_data[1])
fd.interpolation = SplineInterpolation()

functions = np.zeros((num_intervals, num_points))

for i in range(num_intervals):
    interval = np.linspace(from_day+(i*period_length), from_day+(i+1)*period_length, num_points)
    functions[i] = np.squeeze(fd(interval))


fd_intervals = skfda.FDataGrid(functions)
basis_fd_intervals = fd_intervals.to_basis(BSplineBasis(n_basis=8, order=4))



### Apply FPCA

n_comps = 3
comps = np.arange(1, n_comps+1)

fpca_discretized = FPCA(n_components=n_comps)
scores = fpca_discretized.fit_transform(basis_fd_intervals)


scree = fpca_discretized.explained_variance_ratio_
scree = np.array([scree, np.cumsum(scree)])

print('FPC weights: FPC1 = '+str(scree[0][0])+', FPC2 = '+str(scree[0][1])+', FPC3 = '+str(scree[0][2]))


fd_scores = skfda.FDataGrid(data_matrix=scores[:,0:2], grid_points=[1, 2])
mahalanobis = MahalanobisDistance(2, weights=1/scree[0,:2])
mahalanobis.fit(fd_scores)

mean_scores = fd_scores.mean()



# Make scree plot

plt.figure(figsize=(6,4))
p = plt.bar(comps, scree[0])
plt.bar_label(p, label_type='center')
plt.plot(comps, scree[1], marker='o', color='g', label='Cumulative')
plt.xticks(comps, labels=['FPC 1', 'FPC 2', 'FPC 3'])
plt.ylabel('% variation described')
plt.title('Scree Plot')
plt.legend()


# Make plot of FPCs

plt.figure()
plt.plot(np.linspace(0, 1, 1000), np.squeeze(fpca_discretized.components_[0](np.linspace(0, 1, 1000))), label='FPC1')
plt.plot(np.linspace(0, 1, 1000), np.squeeze(fpca_discretized.components_[1](np.linspace(0, 1, 1000))), label='FPC2')
plt.plot(np.linspace(0, 1, 1000), np.squeeze(fpca_discretized.components_[2](np.linspace(0, 1, 1000))), label='FPC3')
plt.axhline(0, color='grey', linestyle='--')
plt.xlim([0,1])
plt.title('Functional Principal Components')
plt.legend()


# Plot FPCA scores

plt.figure()
plt.scatter(scores[:,0], scores[:,1])
plt.xlabel('FPC 1')
plt.ylabel('FPC 2')
plt.title('FPCA scores')




# Plot Mahalanobis distance

heatmap = np.zeros(np.shape(scores)[0])

for i, score in enumerate(fd_scores):
    heatmap[i] = mahalanobis(fd_scores[i], mean_scores)[0]

min1 = 1.1*scores[:,0].min()
max1 = 1.1*scores[:,0].max()
min2 = 1.1*scores[:,1].min()
max2 = 1.1*scores[:,1].max()

xx, yy = np.meshgrid(np.linspace(min1, max1, 100), np.linspace(min2, max2, 100))
contours = np.zeros_like(xx)

for i in range(100):
    for j in range(100):
        temp = [xx[i,j], yy[i,j]]
        contours[i,j] = mahalanobis(skfda.FDataGrid(data_matrix=temp, grid_points=[1, 2]), mean_scores)[0]


plt.figure()
cax = plt.scatter(scores[:,0], scores[:,1], c=heatmap)
plt.contour(np.linspace(min1, max1, 100), np.linspace(min2, max2, 100), contours, levels=np.linspace(0, 1.1*heatmap.max(), 10))
plt.colorbar(cax)
plt.xlabel('FPC 1')
plt.ylabel('FPC 2')
plt.xlim(min1, max1)
plt.ylim([min2, max2])
plt.title('Score plot with Mahalanobis distance to origin')


# Plot distribution of distance

plt.figure()
plt.hist(heatmap, bins=20, density=True)
plt.xlabel('Mahalanobis distance')
plt.ylabel('Occurrence')
plt.title('Mahalanobis distance distribution')



# Find and plot outliers

num_outliers = int(num_intervals*0.05)+1

outliers = np.argsort(heatmap)[-num_outliers:]
#outliers = np.argwhere(heatmap>0.6)[:,0]


plt.figure(figsize=(12,4))
plt.plot(loc_data[0], loc_data[1], '.-', linewidth=0.5)
for i in outliers:
    interval = np.linspace(from_day+(i*period_length), from_day+(i+1)*period_length, num_points)
    plt.axvline(from_day+(i*period_length), color='grey', linestyle='--')
    plt.axvline(from_day+((i+1)*period_length), color='grey', linestyle='--')
plt.xticks(loc_days[::20], labels=loc_dates[::20], rotation=70)
plt.xlim([show_min, show_max])
plt.title(loc)



for i in outliers:
    plt.figure()
    plt.plot(loc_data[0], loc_data[1], '.-', linewidth=0.5)
    plt.axvline(from_day+(i*period_length), color='grey', linestyle='--', label='Time interval')
    plt.axvline(from_day+((i+1)*period_length), color='grey', linestyle='--')
    plt.xticks(loc_days[::2], labels=loc_dates[::2], rotation=70)
    plt.xlim([from_day+((i-4)*period_length), from_day+((i+4)*period_length)])
    plt.title('Outlier in measured viral load - '+loc)
    plt.legend()


outliers = [int(from_day+outlier*period_length) for outlier in outliers]
outliers_dates = [day_index_to_date(outlier, start_date) for outlier in outliers]


print('Outliers:')
print(outliers_dates)


plt.show()


