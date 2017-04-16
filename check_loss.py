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

with open("plot_names.txt") as name_file:
  lines = [line.rstrip() for line in name_file]
  #print (lines)

lines = filter(None, lines)
names_list = []
for line in lines:
  line = line.split()
  line[1:] = [' '.join(line[1:])]
  #print(line)
  names_list.append(line)

index = 1

for i, pclplot in enumerate(pcl_toplot):
  df = pandas.read_pickle(pclplot)
  df_new = df[['TrainingLoss','ValidationLoss']]

  for j, names in enumerate(names_list):
    if names[0] in pclplot:
        #df_new = df_new.rename(columns={'TrainingLoss':'TrainingLoss - ' + names[1]})
        #df_new = df_new.rename(columns={'ValidationLoss':'MeanValidationLoss - ' + names[1]})
        #print(df_new.at[0, 'ValidationLoss'])
        mean_loss = sum(df_new.at[index, 'ValidationLoss'])/(len(df_new.at[index, 'ValidationLoss']))
        #print(mean_loss)
        plt.plot(df_new.at[index, 'TrainingLoss'], label = 'TrainingLoss - ' + names[1])
        plt.plot(range(len(df_new.at[index, 'TrainingLoss'])), [mean_loss]*len(df_new.at[index, 'TrainingLoss']), linestyle='--', label='MeanValidationLoss - ' + names[1])

#plt.plot(df_new.at[0, 'ValidationLoss'])
#plt.plot(df_new.at[0, 'ValidationLoss'])
plt.title('Training and Mean Validation Cost')
plt.xlabel('Computational Steps')
plt.ylabel('Cost')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('test.eps', bbox_inches='tight')
plt.show()

'''
f2 = plt.figure(2)
plt.plot(df_new.at[0, 'ValidationLoss'])
f2.show()
'''
#raw_input()

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
