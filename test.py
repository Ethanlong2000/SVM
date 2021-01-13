STARTNUM=0
ROW1=3
ROW2=5
STOPNUM=10
ma=range(0,10)
list_y={}
for i in range(STARTNUM, ROW1):
    list_y[i-STARTNUM]=ma[i]

for j in range(ROW2, STOPNUM):
    list_y[j-ROW2+ROW1-STARTNUM]=ma[j]
print(list_y)