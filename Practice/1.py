def cal(i,r,min_rs,min_r):
    if i==n :
        if r < min_r[0]:
            min_r[0]=r
            min_rs.clear()
            for item in rs :
                min_rs.append(item)
        return
    r=r*int(s[i])
    rs.append('*')
    rs.append(s[i])
    cal(i+1,r,min_rs,min_r)
    rs.pop()
    rs.pop()
    r=r/int(s[i])
    r=r + int(s[i])
    rs.append('+')
    rs.append(s[i])
    cal(i+1,r,min_rs,min_r)
    rs.pop()
    rs.pop()
    r=r - int(s[i])
n = int(input())
s=input()
flag=True
min_rs=[]
min_r=[]
min_r.append(9**50)
rs=[]
rs.append(s[0])
r=int(s[0])
for item in s:
    if item=='0':
        flag=False
if not flag:
    for item in s :
        min_rs.append(item)
        min_rs.append('*')
    min_rs.pop()
else:
    cal(1,r,min_rs,min_r)
    # print(min_rs)
for x in min_rs:
    print(x,end=(''))