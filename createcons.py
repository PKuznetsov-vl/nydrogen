import itertools
import time

import pandas as pd


# to notebbok

# consumption_train_norm = np.load("data/example_nondeterminist_cons_train.npy")[0:1 * 365 * 24]
# consumption_train_norm=consumption_train_norm*2.1
# print(consumption_train_norm)
# np.save('data/test.npy',consumption_train_norm)

# df= pd.read_csv('/home/pavel/PycharmProjects/nydrogen/data/US_DOT_ALL_FLIGHT_2015.csv')
#
# df.loc[(df['DISTANCE'] <200) & (df['ORIGIN_AIRPORT']=='PSP')].to_csv('/home/pavel/PycharmProjects/nydrogen/data/psp_flights_2015.csv',index=False)
# df=df.loc[df['MONTH'] ==1]
# graph_df=df.groupby(['MONTH'])['DISTANCE'].agg('sum')/1.55 #перевод  в кг
# print(graph_df)
#
# # join
# # platforms_df = df.groupby('Platform')[['Global_Sales']].sum().join(
# #     df.groupby('Platform')[['Name']].count()
# # )
#
# # platforms_df.sort_values('Global_Sales', ascending=False, inplace=True)
# trace0 = go.Scatter(
#     x=graph_df.index,
#     y=graph_df,
#     name='January'
# )
# data = [trace0]
# layout = {'title': 'quantity of hydrogen needed by day'}
#
# # cоздаем объект Figure и визуализируем его
# fig = go.Figure(data=data, layout=layout)
#
# #iplot(fig, show_link=False,filename='file.html')
# plotly.offline.plot(data, filename='../../file.html')
def edit(value):
    if len(value) == 3:
        value = value[0] + ':' + value[1:]
    else:
        value = value[:2] + ':' + value[2:]
    return value


def zero_val(value: str):
    value = value[:value.find(":")]
    return int(value)


def work_time(function):
    def wrapped(*args):
        start_time = time.perf_counter_ns()
        res = function(*args)
        print(time.perf_counter_ns() - start_time)
        return res

    return wrapped


# print(zero_val('17:32'))
# df = pd.read_csv('/home/pavel/PycharmProjects/nydrogen/data/psp_flights_2015.csv',usecols=['YEAR','MONTH','DAY','SCHEDULED_DEPARTURE','DISTANCE'],dtype={'SCHEDULED_DEPARTURE':int})
# df['HOUR']=df['SCHEDULED_DEPARTURE'].apply(lambda x: edit(str(x))).apply(lambda x:zero_val(x))
# df = df.append({'YEAR': 2015, 'MONTH': 7, 'DAY': 4, 'HOUR': 5, 'SCHEDULED_DEPARTURE': 0, 'DISTANCE': 0},
#                  ignore_index=True)
# df.to_csv('0.csv',index=False)
df = pd.read_csv('examples/MG_two_storages/0.csv')


@work_time
def gen():
    tuple_a = []
    months = list(set(df['MONTH'].values))
    print(months)
    for month in months:
        days = list(set(df.loc[df['MONTH'] == month]['DAY'].values))
        for day in days:
            #print(day)
            pass
            tuple_a.append([month, day])
    print(len(tuple_a))
    return tuple_a


@work_time
def tst():
    tuple_a = []
    months = list(set(df['MONTH'].values))
    # atc res=  (month+1 for month in months)
    res = []
    for month in months:
        days = list(set(df.loc[df['MONTH'] == month]['DAY'].values))
        for day in days:
            res.append((month, day))
        print(res)

print(gen())
# df1=df.copy()
# for  day in generate():
#     print( day)

def ds(df):
   # append 10 month in tuple
    for month in gen():
        # for day in set(list(df['DAY'].values)):
        print('month',month[0])
        print('day',month[1])
        #print(df.loc[(df['DAY']==month[1])]['HOUR'].values)
        for i in range(0, 24, 1):# may be if else
            dd = df.loc[(df['MONTH'] == month[0]) & (df['DAY'] == month[1])]['HOUR'].values

            if i in dd:
                print(i)

            else:
                df = df.append({'YEAR': 2015, 'MONTH': month[0], 'DAY': month[1], 'HOUR': i, 'SCHEDULED_DEPARTURE': 0,
                                'DISTANCE': 0}, ignore_index=True)

    df.to_csv('1.csv', index=False)

# print(df1.head(40))


#ds(df=df)
