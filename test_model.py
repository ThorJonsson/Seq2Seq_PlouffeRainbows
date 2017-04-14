import sys
import pandas
import os
import os.path
import sys
import re
#import seaborn

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

#print(names_list)
#exit()

for i, pclplot in enumerate(pcl_toplot):
  df = pandas.read_pickle(pclplot)
  df_new = df[['TrainingLoss','ValidationLoss']]
  df_new = df_new[:9]

  for j, names in enumerate(names_list):
    #print(j)
    if names[0] in pclplot:
        #print(names[0])
        #print(names[1])
        #print(j)
        #print names
        df_new = df_new.rename(columns={'TrainingLoss':'TrainingLoss - ' + names[1]})
        df_new = df_new.rename(columns={'ValidationLoss':'ValidationLoss - ' + names[1]})

  if i == 0:
    ax = df_new.plot(style=['-','--'])
  else:
    df_new.plot(style=['-', '--'],ax=ax)

ax
ax.grid('on', which='major', axis='x')
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
fig = ax.get_figure()
fig.savefig('testall.png')
