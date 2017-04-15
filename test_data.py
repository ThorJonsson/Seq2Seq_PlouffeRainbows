import pickle
import sys
import pandas

df1 = pandas.read_pickle(sys.argv[1])
df2 = pandas.read_pickle(sys.argv[2])
df3 = pandas.read_pickle(sys.argv[3])
df4 = pandas.read_pickle(sys.argv[4])
df5 = pandas.read_pickle(sys.argv[5])

#with open(sys.argv[1]) as f:
#  train_loss, mean_train_loss, valid_loss, mean_valid_loss = pickle.load(f)
#  print(epoch)
'''
print df.at[4,'Epoch']
print df.at[4,'TrainingLoss']
print df.at[4,'MeanTrainingDuration']
print df.at[4,'ValidationLoss']
print df.at[4,'MeanValidDuration']

print df.at[9,'Epoch']
print df.at[9,'TrainingLoss']
print df.at[9,'MeanTrainingDuration']
print df.at[9,'ValidationLoss']
print df.at[9,'MeanValidDuration']

print df.at[14,'Epoch']
print df.at[14,'TrainingLoss']
print df.at[14,'MeanTrainingDuration']
print df.at[14,'ValidationLoss']
print df.at[14,'MeanValidDuration']

print df.at[19,'Epoch']
print df.at[19,'TrainingLoss']
print df.at[19,'MeanTrainingDuration']
print df.at[19,'ValidationLoss']
print df.at[19,'MeanValidDuration']

#ax = df.plot(style=['o','rx'])
#fig = ax.get_figure()
#fig.savefig('test.png')
'''

df1 = df1[['TrainingLoss', 'ValidationLoss']]
df1 = df1.rename(columns={'TrainingLoss':'TrainingLoss - lr = 0.5'})
df1 = df1.rename(columns={'ValidationLoss':'ValidationLoss - lr = 0.5'})
df1 = df1[:21]
#ax = df1.plot(style=['#a6cee3','#1f78b4'])

df2 = df2[['TrainingLoss', 'ValidationLoss']]
df2 = df2.rename(columns={'TrainingLoss':'TrainingLoss - lr = 0.1'})
df2 = df2.rename(columns={'ValidationLoss':'ValidationLoss - lr = 0.1'})
df2 = df2[:21]
#ax = df2.plot(style=['#b2df8a','#33a02c'])

df3 = df3[['TrainingLoss', 'ValidationLoss']]
df3 = df3.rename(columns={'TrainingLoss':'TrainingLoss - lr = 0.05'})
df3 = df3.rename(columns={'ValidationLoss':'ValidationLoss - lr = 0.05'})
df3 = df3[:21]
ax = df3.plot(style=['#fb9a99', '#e31a1c'], label=['TrainingLoss lr=0.005', 'ValidationLoss lr=0.005'])

df4 = df4[['TrainingLoss', 'ValidationLoss']]
df4 = df4.rename(columns={'TrainingLoss':'TrainingLoss - lr = 0.01'})
df4 = df4.rename(columns={'ValidationLoss':'ValidationLoss - lr = 0.01'})
df4 = df4[:21]
#df4.plot(style=['#fdbf6f', '#ff7f00'], ax=ax)

df5 = df5[['TrainingLoss', 'ValidationLoss']]
df5 = df5.rename(columns={'TrainingLoss':'TrainingLoss - lr = 0.001'})
df5 = df5.rename(columns={'ValidationLoss':'ValidationLoss - lr = 0.001'})
df5 = df5[:21]
#df5.plot(style=['#cab2d6', '#984ea3'], ax=ax)

ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
fig = ax.get_figure()
fig.savefig('testall.png')
