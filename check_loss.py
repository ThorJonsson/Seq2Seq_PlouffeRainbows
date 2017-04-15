from __future__ import division

import sys
import pandas
import os
import os.path
import sys
import re
import seaborn
import matplotlib.pyplot as plt

pcl_files = []
dataframes = []
for dirpath, dirnames, filenames in os.walk("./logs"):
  for filename in [f for f in filenames if f.endswith(".pcl")]:
    print os.path.join(dirpath, filename)
    pcl_files.append(os.path.join(dirpath, filename))

pcl_toplot = []
for pcl in pcl_files:
  #print(pcl)
  #print(sys.argv)
  experiments = set(sys.argv[1:])
  pcl_set = set(pcl.split("/"))
  #print(pcl_set)
  #print(experiments)
  if experiments.intersection(pcl_set):
    #print(pandas.read_pickle(pcl))
    pcl_toplot.append(pcl)
    #exit()

for i, pclplot in enumerate(pcl_toplot):
	df = pandas.read_pickle(pclplot)
 	df_new = df[['TrainingLoss','ValidationLoss']]
 	#df_new = df_new.unstack().apply(pandas.Series)
 	#print(df_new)
 	#print(df_new)
 	print(df_new.at[0, 'TrainingLoss'])

plt.plot(df_new.at[0, 'TrainingLoss'])
plt.plot(df_new.at[1, 'TrainingLoss'])
plt.show()
'''
df_test0 = df_new.at[0, 'TrainingLoss']
df_test1 = df_new.at[1, 'TrainingLoss']
df_test2 = df_new.at[2, 'TrainingLoss']

ax = df_test0.plot()
df_test1.plot(ax=ax)
df_test2.plot(ax=ax)
fig = ax.get_figure()
fig.savefig('testloss.png')
'''