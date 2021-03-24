# 题目：有四个数字：1、2、3、4，能组成多少个互不相同且无重复数字的三位数？各是多少？

num=[1,2,3,4]
for h in num:
    for d in num:
        if d != h :
            for o in num :
                if ((o != d) and (o != h) ):
                    print(h*100+d*10+o)
