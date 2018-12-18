import pandas as pd
import time

col=['a','b','c']

st=time.time()
df=pd.DataFrame(columns=col)

for i in range(1000):
    df.loc[i]=[1,1,1]
end=time.time()
print(end-st)


st=time.time()
all_l=[]
for i in range(1000):
    all_l.append([1,1,1])

df=pd.DataFrame(all_l,columns=col)
end=time.time()
print(end-st)
print(df)