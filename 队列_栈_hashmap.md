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

##### [84. 柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/)

给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

 

**示例 1:**

![img](https://assets.leetcode.com/uploads/2021/01/04/histogram.jpg)

```
输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为图中红色区域，面积为 10
```

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        heights = [0] + heights + [0]#前后分别加一个哨兵，方便使用单调栈
        res = 0
        for i in range(len(heights)):
            #print(stack, heights[i])
            while stack and heights[stack[-1]] > heights[i]: #如果遇见了第一个小于当前栈顶在数组中的值，则需要将栈顶所存储的数组的值的序号弹出栈
                tmp = stack.pop() #将栈顶所存储的数组的值的序号弹出栈，作为高度，这说明这个高度能够跨越的宽度已经到头
                #因为不包括前后两个，只包括中间的 因此需要-1
                print(heights[stack[-1]])
                res = max(res, (i - stack[-1] - 1) * heights[tmp])
            stack.append(i)#根本上述的如果遇见的值大于栈顶的存储的序号在数组中的值，那么就将这个序号压入栈，
            #继续寻找小于栈顶存储序号在数组中的值
        return res
```



```
解题思路

单调栈的意思可以看这个链接，这位大佬写的很详细https://blog.csdn.net/liujian20150808/article/details/50752861 单调栈，可以理解为有单调性的一个数组，在这里我们可以理解成为栈数组，栈数组可以是单调增，也可以是单调减，在本题中我们使用的是单调增 即栈数组从0到数组最后是单调递增的。单调栈的数组性质也是我们常用的性质，到这里为止，我们对单调栈有了基本印象，一个单调递的数组 从栈底到栈顶，单调递增，可以理解为从栈数组的0位置开始到栈数组的最后一位一直是单调递增。 **单调栈的性质：**接下来我们分析单调栈的性质，因为单调栈是即栈数组是单调递增的，因此他的单调性必须一直保持，因此在遇到下一个需要插入 栈数组的值的时候，我们首先判断该值与栈顶元素相比那个大？我们设栈顶元素为a，需要插入的值为b，当a<b的时候，我们可以直接将b插入栈数组 ，我们可以设原来的栈数组为[0 , a],0到a单调递增，符合单调栈的要求 1：a<b的时候 b可以直接压入栈，栈数组变成[0 , a , b] 2：a>b的时候 b无法直接压入栈，因为那样会破坏栈的单调性我们需要将a弹出栈，然后再将b压入栈 栈数组就会变成[0 , b] **扩展-举例：我们扩展来讲，如果原来的栈数组为[0 , 1 , 5 , 8] ，可以看出栈顶元素为8，我们需要将9压入栈的话，因为9>8,所以压入之后 栈数组变成[0 , 1 , 5 , 8 , 9]不影响原来的单调性，依然是单调递增 但是如果我们需要将4压入栈的话，我们需要先将大于4的值弹出栈，整个过程是这样的，先将要压入栈的值与当前的栈顶元素比较，即4与8比较 因为4<8，所以4无法压入栈，因此我们要弹出8，弹出8之后栈数组变成[0 , 1 , 5],然后4继续与栈顶元素5比较，因为4<5，因此我们需要弹出5， 弹出之后栈顶数组变成[0 , 1],然后4与栈顶元素1比较，因为4>1，因此4可以压入栈，压入之后栈数组变成[0 , 1 , 4] 这就是单调栈的思想，把他变成数组来讲就很好理解-单调减的栈反过来理解即可（我想我已经表达的很清楚，不懂的可以在评论里指出来， 我看到会回复解答的）

好的现在我们已经懂了单调栈的思想了，那我们开始来做题： **题目分析：**这道题是求可以勾勒出来的最大矩形的面积，问题简化用一句话概括就是，我们要求出不同高度下的最大矩形面积，选择其中最大的面积 作为我们要输出的答案。这么来看问题已经转化成求每个高度下，对应的最大矩形面积（因为不同高度下都对应着这个高度下的一个最大矩形面积， 我们并不知道这么多个矩形面积，那个是最大的，我们只能求出来每个，然后做比较从而求出最大值==我们要输出的答案） **针对本题的解法：**我们需要依次遍历整个数组 例如：heights = [2,1,5,6,2,3]，首先我们要在数组前后分别加一个0，使得数组变成 heights = [0,2,1,5,6,2,3,0],在这里我们把这两个0成为哨兵，因为如果不加这个0，会导致遍历过后栈中可能会还存有元素，需要再加一步讨论 如果加了的话，就不用讨论了，相当于优化了过程。（可以理解成，如果最后一个3入栈之后，如何将3弹出呢？只有遇见0，0<3,如果压0入栈 才能将3弹出，这样我们才能保证我们有用的元素都弹出栈）这是加哨兵的原因。 在我们遍历数组的时候，假设我们遍历的是数组的第 i 个元素，对应的数组的值是 heights[i],我们将这个值与栈数组的栈顶元素比较，如果 大于栈顶元素，则压入栈成为新的栈顶元素，然后继续下一个，直到找到小于栈顶元素的那个值，假设这个是数组的第j个元素，然后栈顶元素为 数组的第a个元素，因为heights[a]>heights[j],因此为了不破坏单调栈的单调性，我们需要将heights[a]弹出栈数组， 那么我们可以得到数组的第a个元素的对应的最大矩形面积为heigh[a](j-a-1),因为只取中间的值，所以需要j-a-1，自己画图看看就知道了 然后我们不停的遍历数组，设一个初始值为ans = 0 ，然后 ans = max(ans , heigh[a](j-a-1))，不停的遍历即可，最终输出ans， 即为我们想要的结果。 为什么我们遇到比当前矩阵高度小的数，才需要弹出栈然后求最大矩形面积呢？ 问题在于这里，因为举例，对于heights = [2,1,5,6,2,3]来说，    

比如说一开始是2，如果遇到下一个是1，那么这个时候高度为2的矩阵的最大宽度就已经达到了，因为在矩阵高度一定的情况下，必然是宽度越宽 面积越大，因此我们在保证矩阵高度的情况下，遇到第一个小于我们规定的矩阵高度的值，即遇到了我们可以拓展的最大的矩阵宽度， 即为高度小于规定矩阵高度的上一个值。这样就是我们用单调栈的意义，他可以灵敏的检测到第一个小于我们规定值的值。

```

##### [218. 天际线问题](https://leetcode.cn/problems/the-skyline-problem/)

城市的 **天际线** 是从远处观看该城市中所有建筑物形成的轮廓的外部轮廓。给你所有建筑物的位置和高度，请返回 *由这些建筑物形成的 **天际线*** 。

每个建筑物的几何信息由数组 `buildings` 表示，其中三元组 `buildings[i] = [lefti, righti, heighti]` 表示：

- `lefti` 是第 `i` 座建筑物左边缘的 `x` 坐标。
- `righti` 是第 `i` 座建筑物右边缘的 `x` 坐标。
- `heighti` 是第 `i` 座建筑物的高度。

你可以假设所有的建筑都是完美的长方形，在高度为 `0` 的绝对平坦的表面上。

**天际线** 应该表示为由 “关键点” 组成的列表，格式 `[[x1,y1],[x2,y2],...]` ，并按 **x 坐标** 进行 **排序** 。**关键点是水平线段的左端点**。列表中最后一个点是最右侧建筑物的终点，`y` 坐标始终为 `0` ，仅用于标记天际线的终点。此外，任何两个相邻建筑物之间的地面都应被视为天际线轮廓的一部分。

**注意：**输出天际线中不得有连续的相同高度的水平线。例如 `[...[2 3], [4 5], [7 5], [11 5], [12 7]...]` 是不正确的答案；三条高度为 5 的线应该在最终输出中合并为一个：`[...[2 3], [4 5], [12 7], ...]` 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/12/01/merged.jpg)

```
输入：buildings = [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]
输出：[[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]
解释：
图 A 显示输入的所有建筑物的位置和高度，
图 B 显示由这些建筑物形成的天际线。图 B 中的红点表示输出列表中的关键点。
```

```python
class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        import heapq
        # 将所有端点加入到点集中（每个建筑物的左右端点）
        events = sorted([[l, -H, r] for l, r, H in buildings] + [[r, 0, 0] for _, r, _ in buildings])
        res = [[0, 0]] 
        heap = [[0, float('inf')]]  # 哨兵，当堆中不含有任何建筑物，天际线是0
        for l, H, r in events:
            # 出堆：保证当前堆顶为去除之前建筑物右端点的最大值
            while l >= heap[0][1]:
                heapq.heappop(heap)
            # 入堆：所有建筑从左端点开始入堆（这里可以把右端点当成高度为0的建筑的左端点）
            if H:
                heapq.heappush(heap, [H, r])
            # 关键点：判断是否是关键点
            if res[-1][1] != -heap[0][0]:
                res.append([l, -heap[0][0]])
        return res[1:]
```

方法二

![image-20231028143914947](/Users/kaixiwu/Library/Application Support/typora-user-images/image-20231028143914947.png)

```python
class Solution:
    def getSkyline(self, buildings):
        if not buildings: return []
        if len(buildings) == 1:
            return [[buildings[0][0], buildings[0][2]], [buildings[0][1], 0]]
        mid = len(buildings) // 2
        left = self.getSkyline(buildings[:mid])
        right = self.getSkyline(buildings[mid:])
        return self.merge(left, right)

    # 两个合并
    def merge(self, left, right):
        # 记录目前左右建筑物的高度
        lheight = rheight = 0
        # 位置
        l = r = 0
        # 输出结果
        res = []
        while l < len(left) and r < len(right):
            if left[l][0] < right[r][0]:
                # current point
                cp = [left[l][0], max(left[l][1], rheight)]
                lheight = left[l][1]
                l += 1
            elif left[l][0] > right[r][0]:
                cp = [right[r][0], max(right[r][1], lheight)]
                rheight = right[r][1]
                r += 1
            # 相等情况
            else:
                cp = [left[l][0], max(left[l][1], right[r][1])]
                lheight = left[l][1]
                rheight = right[r][1]
                l += 1
                r += 1
            # 和前面高度比较，不一样才加入
            if len(res) == 0 or res[-1][1] != cp[1]:
                res.append(cp)
        # 剩余部分添加进去
        res.extend(left[l:] or right[r:])
        return res

```



##### [295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/)

**中位数**是有序整数列表中的中间值。如果列表的大小是偶数，则没有中间值，中位数是两个中间值的平均值。

- 例如 `arr = [2,3,4]` 的中位数是 `3` 。
- 例如 `arr = [2,3]` 的中位数是 `(2 + 3) / 2 = 2.5` 。

实现 MedianFinder 类:

- `MedianFinder() `初始化 `MedianFinder` 对象。
- `void addNum(int num)` 将数据流中的整数 `num` 添加到数据结构中。
- `double findMedian()` 返回到目前为止所有元素的中位数。与实际答案相差 `10-5` 以内的答案将被接受。

**示例 1：**

```
输入
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
输出
[null, null, null, 1.5, null, 2.0]

解释
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // 返回 1.5 ((1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0
```



![image-20231028160501022](/Users/kaixiwu/Library/Application Support/typora-user-images/image-20231028160501022.png)



```python

class MedianFinder:
    
    def __init__(self):
        self.left, self.right = [], [] #双heap找中位数


    def addNum(self, num: int) -> None:
        if len(self.left)==len(self.right):
            heapq.heappush(self.left, -heapq.heappushpop(self.right, num)) #right的最小值放入left
        else:
            heapq.heappush(self.right, -heapq.heappushpop(self.left, -num))#left的最大值放入right
            # 通过left和right长度一致来保证left都是小于中位数的，right都是大于中位数


    def findMedian(self) -> float:
        if len(self.left)==len(self.right):
            return (-self.left[0]+self.right[0])/2
        else:
            return -self.left[0]



# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```

