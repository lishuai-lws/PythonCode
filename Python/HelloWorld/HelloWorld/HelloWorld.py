print('Hello World')
a=10
print(a,type(a),id(a))
print(ord('1'))
print(chr(64))
print(bool(False))
print(bool(0))
print(bool(0.0))
print(bool(None))
print(bool(''))
print(bool([]))
print(bool(list()))
print(bool(tuple()))
# a = float(input('please enter a number :'))
# print(a)
r = range(10)
print(r,type(r))
sum =0
a=0
while a<=100 :
    if a%2 == 0:
        sum+=a
    a+=1
print('和为：',sum)
def fun(a,b,c):
    global age
    print(a)
    print(b)
    print(c)
l=[10,20,30]
d={'a':10,'b':20,'c':30}
fun(*l)
fun(**d)
fun(10,20,30)