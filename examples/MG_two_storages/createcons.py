import numpy as np
import  pandas as pd
import plotly

import plotly.graph_objs as go
from plotly.offline import iplot

# consumption_train_norm = np.load("data/example_nondeterminist_cons_train.npy")[0:1 * 365 * 24]
# consumption_train_norm=consumption_train_norm*2.1
# print(consumption_train_norm)
# np.save('data/test.npy',consumption_train_norm)

df= pd.read_csv('/home/pavel/PycharmProjects/nydrogen/data/tst.csv')

df.loc[df['DISTANCE'] <100].to_csv('/home/pavel/PycharmProjects/nydrogen/data/tst.csv')
#df=df.loc[df['MONTH'] ==1]
graph_df=df.groupby(['DAY'])['DISTANCE'].agg('sum')/1.55


# считаем число проданных и вышедших игр по платформам
# platforms_df = df.groupby('Platform')[['Global_Sales']].sum().join(
#     df.groupby('Platform')[['Name']].count()
# )

# platforms_df.sort_values('Global_Sales', ascending=False, inplace=True)
trace0 = go.Scatter(
    x=graph_df.index,
    y=graph_df,
    name='January'
)
data = [trace0]
layout = {'title': 'quantity of hydrogen needed by day'}

# cоздаем объект Figure и визуализируем его
fig = go.Figure(data=data, layout=layout)

#iplot(fig, show_link=False,filename='file.html')
plotly.offline.plot(data, filename='../../file.html')