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

**示例 1:**

```
输入：
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

输出：
[null,null,null,null,-3,null,0,-2]

解释：
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.
```

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



##### [232. 用栈实现队列](https://leetcode.cn/problems/implement-queue-using-stacks/)

请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（`push`、`pop`、`peek`、`empty`）：

实现 `MyQueue` 类：

- `void push(int x)` 将元素 x 推到队列的末尾
- `int pop()` 从队列的开头移除并返回元素
- `int peek()` 返回队列开头的元素
- `boolean empty()` 如果队列为空，返回 `true` ；否则，返回 `false`

**说明：**

- 你 **只能** 使用标准的栈操作 —— 也就是只有 `push to top`, `peek/pop from top`, `size`, 和 `is empty` 操作是合法的。
- 你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。



```python 
class MyQueue:

    def __init__(self):
        self.q = deque()


    def push(self, x: int) -> None:
        self.q.append(x)


    def pop(self) -> int:
        return self.q.popleft()


    def peek(self) -> int:
        return self.q[0]


    def empty(self) -> bool:
        return len(self.q)==0



# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
```



##### [20. 有效的括号](https://leetcode.cn/problems/valid-parentheses/)

给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串 `s` ，判断字符串是否有效。

有效字符串需满足：

1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。
3. 每个右括号都有一个对应的相同类型的左括号。

**示例 2：**

```
输入：s = "()[]{}"
输出：true
```

```python
class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) % 2 == 1:
            return False
        pairs = {")":"(", "}":"{", "]":"["}
        stack = deque() # 比list快很多
        for ch in s:
            if ch in pairs:
                if not stack or pairs[ch]!= stack[-1]:
                    return False
                stack.pop()
            else:
                stack.append(ch)
        
        return not stack
```

##### [150. 逆波兰表达式求值](https://leetcode.cn/problems/evaluate-reverse-polish-notation/)

给你一个字符串数组 `tokens` ，表示一个根据 [逆波兰表示法](https://baike.baidu.com/item/逆波兰式/128437) 表示的算术表达式。

请你计算该表达式。返回一个表示表达式值的整数。

**注意：**

- 有效的算符为 `'+'`、`'-'`、`'*'` 和 `'/'` 。
- 每个操作数（运算对象）都可以是一个整数或者另一个表达式。
- 两个整数之间的除法总是 **向零截断** 。
- 表达式中不含除零运算。
- 输入是一个根据逆波兰表示法表示的算术表达式。
- 答案及所有中间计算结果可以用 **32 位** 整数表示。

**示例 2：**

```
输入：tokens = ["4","13","5","/","+"]
输出：6
解释：该算式转化为常见的中缀算术表达式为：(4 + (13 / 5)) = 6
```

**示例 3：**

```
输入：tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
输出：22
解释：该算式转化为常见的中缀算术表达式为：
  ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
```

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        q = []
        for s in tokens:
            if s in ["+", "-", "*", "/"]:
                d1 = q.pop()
                d2 = q.pop()
                if s == "+":
                    d = d1+d2
                elif s == "-":
                    d = d2-d1
                elif s == "/":
                    d = int(d2/d1)
                else:
                    d = d2*d1
                q.append(d)
            else:
                q.append(int(s))
                #d = int(s)

        return q[-1]
```

##### [227. 基本计算器 II](https://leetcode.cn/problems/basic-calculator-ii/)

给你一个字符串表达式 `s` ，请你实现一个基本计算器来计算并返回它的值。

整数除法仅保留整数部分。

你可以假设给定的表达式总是有效的。所有中间结果将在 `[-231, 231 - 1]` 的范围内。

**注意：**不允许使用任何将字符串作为数学表达式计算的内置函数，比如 `eval()` 。

**示例 1：**

```
输入：s = "3+2*2"
输出：7
```

```python
class Solution:
    def calculate(self, s):
        stack = []
        pre_op = '+' #初始化运算符
        num = 0
        for i, each in enumerate(s):
            if each.isdigit():
                num = 10 * num + int(each) # for example "3+2*2*4+55+66"
            if i == len(s) - 1 or each in '+-*/':
                if pre_op == '+':
                    stack.append(num)
                elif pre_op == '-':
                    stack.append(-num)
                elif pre_op == '*':
                    stack.append(stack.pop() * num)
                elif pre_op == '/':
                    top = stack.pop()
                    if top < 0:
                        stack.append(int(top / num))
                    else:
                        stack.append(top // num)
                pre_op = each
                num = 0
        return sum(stack) # 将加减乘除全部转换为正负号加减
```



##### [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回 *滑动窗口中的最大值* 。

**示例 1：**

```
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        deque = collections.deque()
        res, n = [], len(nums)
        for i, j in zip(range(1 - k, n + 1 - k), range(n)):
            # 删除 deque 中对应的 nums[i-1]
            if i > 0 and deque[0] == nums[i - 1]:
                deque.popleft()
            # 保持 deque 递减
            while deque and deque[-1] < nums[j]:
                deque.pop()
            deque.append(nums[j])
            # 记录窗口最大值
            if i >= 0:
                res.append(deque[0])
        return res

      
#方法2
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        # 注意 Python 默认的优先队列是小根堆
        q = [(-nums[i], i) for i in range(k)]
        heapq.heapify(q)

        ans = [-q[0][0]]
        for i in range(k, n):
            heapq.heappush(q, (-nums[i], i))
            while q[0][1] <= i - k:
                heapq.heappop(q)
            ans.append(-q[0][0])
        
        return ans

```

