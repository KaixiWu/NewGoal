## Queue题目

##### [225. 用队列实现栈](https://leetcode.cn/problems/implement-stack-using-queues/)

请你仅使用两个队列实现一个后入先出（LIFO）的栈，并支持普通栈的全部四种操作（push、top、pop 和 empty）。

实现 MyStack 类：

void push(int x) 将元素 x 压入栈顶。
int pop() 移除并返回栈顶元素。
int top() 返回栈顶元素。
boolean empty() 如果栈是空的，返回 true ；否则，返回 false 。


注意：

你只能使用队列的基本操作 —— 也就是 push to back、peek/pop from front、size 和 is empty 这些操作。
你所使用的语言也许不支持队列。 你可以使用 list （列表）或者 deque（双端队列）来模拟一个队列 , 只要是标准的队列操作即可。



```python
class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue = collections.deque()


    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        n = len(self.queue)
        self.queue.append(x)
        for _ in range(n):
            self.queue.append(self.queue.popleft())


    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        return self.queue.popleft()


    def top(self) -> int:
        """
        Get the top element.
        """
        return self.queue[0]


    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return not self.queue

```



##### [346. 数据流中的移动平均值](https://leetcode.cn/problems/moving-average-from-data-stream/)

给定一个整数数据流和一个窗口大小，根据该滑动窗口的大小，计算其所有整数的移动平均值。

实现 MovingAverage 类：

MovingAverage(int size) 用窗口大小 size 初始化对象。
double next(int val) 计算并返回数据流中最后 size 个值的移动平均值。

```python
class MovingAverage:

    def __init__(self, size: int):
        self.size = size
        self.q = deque()
        self.sum = 0


    def next(self, val: int) -> float:
        if len(self.q) == self.size:
            self.sum -= self.q.popleft()
        self.sum += val
        self.q.append(val)
        return self.sum/len(self.q)
```



##### [1429. 第一个唯一数字](https://leetcode.cn/problems/first-unique-number/)

给定一系列整数，插入一个队列中，找出队列中第一个唯一整数。

实现 FirstUnique 类：

FirstUnique(int[] nums) 用数组里的数字初始化队列。
int showFirstUnique() 返回队列中的 第一个唯一 整数的值。如果没有唯一整数，返回 -1。（译者注：此方法不移除队列中的任何元素）
void add(int value) 将 value 插入队列中。

```python
class FirstUnique:
    # 哈希数据结构

    def __init__(self, nums: List[int]):
        # 1.明确队列需要维护的数据特征(此题为无重复出现的数据) 
        # 2.巧妙利用Counter函数记录出现次数
        self.rec = deque() # 记录无重复数组 #deque() 数据结构比[] 快很多
        self.numsdict = Counter(nums) # Counter函数也是内置函数
        for key in nums:
            if self.numsdict[key]==1:
                self.rec.append(key)

    def showFirstUnique(self) -> int:
        if len(self.rec)>0:
            return self.rec[0]
        else:
            return -1

    def add(self, value: int) -> None:
        self.numsdict[value] += 1
        if self.numsdict[value] == 1:
            self.rec.append(value)
        while len(self.rec)>0 and self.numsdict[self.rec[0]] > 1: #比remove快很多， 
            #利用queue先进先出，后进后出的概念，维护queue的第一个元素一定是唯一的
            self.rec.popleft()
        #else:
        #    if self.numsdict[value]==2: # 如果value已经在rec中， 则把rec中的value去除
        #        self.rec.remove(value)
```



##### [54. 螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/)

给你一个 `m` 行 `n` 列的矩阵 `matrix` ，请按照 **顺时针螺旋顺序** ，返回矩阵中的所有元素。

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res = []
        top, bottom, left, right = 0, len(matrix)-1, 0, len(matrix[0])-1
        while left <= right and top <= bottom:
            # 一个while循环内遍历完上右下左，再更新top,bottom,left,right
            for i in range(left, right+1):
                res.append(matrix[top][i])# 遍历上
            for j in range(top+1, bottom+1):
                res.append(matrix[j][right])# 遍历右
            #先遍历上和右
            if left < right and top < bottom: #同时满足此条件，才遍历下和左， 否则会导致重复遍历
                for i in range(right-1, left-1, -1):
                    res.append(matrix[bottom][i]) #遍历下
                for j in range(bottom-1, top, -1):
                    res.append(matrix[j][left]) #遍历左
            top, bottom, left, right = top+1, bottom-1, left+1, right-1
        return res
```



##### [362. 敲击计数器](https://leetcode.cn/problems/design-hit-counter/)

(低频)

```python
class HitCounter:

    def __init__(self):
        self.rec =[]


    def hit(self, timestamp: int) -> None:
        self.rec.append(timestamp)


    def getHits(self, timestamp: int) -> int:
        res = 0
        for hit in self.rec:
            if timestamp - hit <300:
                res+=1
        return res
```



## Stack题目

##### [155. 最小栈](https://leetcode.cn/problems/min-stack/)

设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。

实现 MinStack 类:

MinStack() 初始化堆栈对象。
void push(int val) 将元素val推入堆栈。
void pop() 删除堆栈顶部的元素。
int top() 获取堆栈顶部的元素。
int getMin() 获取堆栈中的最小元素。

```python
class MinStack:

    def __init__(self):
        self.stack = []
        self.minstack = []


    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.minstack:
            self.minstack.append(val)
        else:
            self.minstack.append(min(val, self.minstack[-1])) #维护最小元素


    def pop(self) -> None:
        self.stack.pop()
        self.minstack.pop()


    def top(self) -> int:
        return self.stack[-1]


    def getMin(self) -> int:
        return self.minstack[-1]
```

