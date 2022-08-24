import functools
# 偏函数
def test(a, b, c, d=1):
    print(a + b + c + d)

def test2(a, b, c, d=2):
    print(a + b + c + d)


newFunc = functools.partial(test, c=5)
newFunc(1, 2)

class Money():
    pass
