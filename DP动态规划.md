[198. 打家劫舍](https://leetcode.cn/problems/house-robber/)

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警**。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **不触动警报装置的情况下** ，一夜之内能够偷窃到的最高金额。

 

**示例 1：**

```
输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。
```

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums)==1:
            return nums[0]
        res = []
        res.append(nums[0])
        res.append(max(nums[0], nums[1]))
        for i in range(2, len(nums)):
            res.append(max(res[i-1], res[i-2]+nums[i])) # 理解逻辑
        return res[-1]
```

[53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/)

给你一个整数数组 `nums` ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**子数组** 是数组中的一个连续部分。

**示例 1：**

```
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
```

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        summin = min(0, nums[0])
        S = nums[0]
        res = nums[0]
        for i in range(1, len(nums)):
            summin = min(S, summin)
            S += nums[i]
            res = max(res, S-summin) 
        return res
```



##### [122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)

给你一个整数数组 `prices` ，其中 `prices[i]` 表示某支股票第 `i` 天的价格。

在每一天，你可以决定是否购买和/或出售股票。你在任何时候 **最多** 只能持有 **一股** 股票。你也可以先购买，然后在 **同一天** 出售。

返回 *你能获得的 **最大** 利润* 。

 

**示例 1：**

```
输入：prices = [7,1,5,3,6,4]
输出：7
解释：在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5 - 1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6 - 3 = 3 。
     总利润为 4 + 3 = 7 。
```

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) <= 1:
            return 0
        res = 0
        for i in range(1, len(prices)):
            cur = max(0, prices[i] - prices[i-1])
            res += cur
        return res
```

##### [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

给定一个数组 `prices` ，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。

你只能选择 **某一天** 买入这只股票，并选择在 **未来的某一个不同的日子** 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 `0` 。

 

**示例 1：**

```
输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
```

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        res = 0
        cur = 0
        for i in range(1, len(prices)):
            profit = prices[i] - prices[i-1]
            cur = max(0, cur+profit)
            res = max(cur, res)
        return res
```



##### [55. 跳跃游戏](https://leetcode.cn/problems/jump-game/)

给你一个非负整数数组 `nums` ，你最初位于数组的 **第一个下标** 。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标，如果可以，返回 `true` ；否则，返回 `false` 。

**示例 1：**

```
输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
```

![image-20231019003219119](/Users/kaixiwu/Library/Application Support/typora-user-images/image-20231019003219119.png)

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n, rightmost = len(nums), 0
        for i in range(n):
            if i <= rightmost:
                rightmost = max(rightmost, i + nums[i])
                if rightmost >= n - 1:
                    return True
        return False
```



##### [62. 不同路径](https://leetcode.cn/problems/unique-paths/)

一个机器人位于一个 `m x n` 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

**示例 1：**

![img](https://pic.leetcode.cn/1697422740-adxmsI-image.png)

```
输入：m = 3, n = 7
输出：28
```

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        return comb(m+n-2, n-1)
      
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1]*n] + [[1]+[0] * (n-1) for _ in range(m-1)]
        #print(dp)
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]

```

##### [322. 零钱兑换](https://leetcode.cn/problems/coin-change/)

给你一个整数数组 `coins` ，表示不同面额的硬币；以及一个整数 `amount` ，表示总金额。

计算并返回可以凑成总金额所需的 **最少的硬币个数** 。如果没有任何一种硬币组合能组成总金额，返回 `-1` 。

你可以认为每种硬币的数量是无限的。

**示例 1：**

```
输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
```

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # 典型dp，要找到传递函数
        # dp[i] = min(dp[i], dp[i-coin] + 1) 遍历所有可能的coin，且i要大于等于coin
        # 因为要求dp[i-coin]的i-coin要大于0，所以外层循环coin，内层循环i-coin到amount

        dp = [float('inf')] * (amount + 1)
        dp[0] = 0

        for coin in coins:
            for x in range(coin, amount+1):
                dp[x] = min(dp[x], dp[x-coin]+1)

        return dp[amount] if dp[amount] != float('inf') else -1

"""
复杂度分析

时间复杂度：O(Sn)，其中 S 是金额，n 是面额数。我们一共需要计算 O(S) 个状态，S 为题目所给的总金额。对于每个状态，每次需要枚举 n 个面额来转移状态，所以一共需要 O(Sn) 的时间复杂度。
空间复杂度：O(S)。数组 dp 需要开长度为总金额 S 的空间。


"""
```





##### [300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)

给你一个整数数组 `nums` ，找到其中最长严格递增子序列的长度。

**子序列** 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3,6,2,7]` 是数组 `[0,3,1,6,2,2,7]` 的子序列。

**示例 1：**

```
输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
```

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        dp = []
        for i in range(len(nums)):
            dp.append(1)
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp)
```

##### [131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/)

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是 **回文串** 。返回 `s` 所有可能的分割方案。

**回文串** 是正着读和反着读都一样的字符串。

**示例 1：**

```
输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]
```

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)
        f = [[True] * n for _ in range(n)]
        
        #预处理回文串
        for i in range(n-1, -1, -1):
            for j in range(i+1, n):
                f[i][j] = (s[i] == s[j]) and f[i+1][j-1]
        
        ret = list()
        ans = list()

        def dfs(i):
            if i == n:
                ret.append(ans[:])
                return 
            for j in range(i, n):
                if f[i][j]:
                    ans.append(s[i:j+1])
                    dfs(j+1)
                    ans.pop()
        dfs(0)
        return ret
```

![image-20231024223945860](/Users/kaixiwu/Library/Application Support/typora-user-images/image-20231024223945860.png)

![image-20231024223958819](/Users/kaixiwu/Library/Application Support/typora-user-images/image-20231024223958819.png)



##### [44. 通配符匹配](https://leetcode.cn/problems/wildcard-matching/)

给你一个输入字符串 (`s`) 和一个字符模式 (`p`) ，请你实现一个支持 `'?'` 和 `'*'` 匹配规则的通配符匹配：

- `'?'` 可以匹配任何单个字符。
- `'*'` 可以匹配任意字符序列（包括空字符序列）。

判定匹配成功的充要条件是：字符模式必须能够 **完全匹配** 输入字符串（而不是部分匹配）。

**示例 1：**

```
输入：s = "aa", p = "a"
输出：false
解释："a" 无法匹配 "aa" 整个字符串。
```

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)

        dp = [[False] * (n+1) for _ in range(m+1)]
        dp[0][0] = True
        for i in range(1, n+1):
            if p[i-1] == "*":
                dp[0][i] = True
            else:
                break
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                if p[j-1] == "*":
                    dp[i][j] = dp[i][j-1] or dp[i-1][j]
                elif p[j-1] == "?" or s[i-1] == p[j-1]:
                    dp[i][j] = dp[i-1][j-1]
        return dp[m][n]
```

![image-20231025001653695](/Users/kaixiwu/Library/Application Support/typora-user-images/image-20231025001653695.png)

![image-20231025001743489](/Users/kaixiwu/Library/Application Support/typora-user-images/image-20231025001743489.png)

##### [10. 正则表达式匹配](https://leetcode.cn/problems/regular-expression-matching/)

给你一个字符串 `s` 和一个字符规律 `p`，请你来实现一个支持 `'.'` 和 `'*'` 的正则表达式匹配。

- `'.'` 匹配任意单个字符
- `'*'` 匹配零个或多个前面的那一个元素

所谓匹配，是要涵盖 **整个** 字符串 `s`的，而不是部分字符串。

 **示例 1：**

```
输入：s = "aa", p = "a"
输出：false
解释："a" 无法匹配 "aa" 整个字符串
```

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n+1) for _ in range(m+1)]
        
        # 初始化
        dp[0][0] = True
        for j in range(1, n+1):
            if p[j-1] == '*':
                dp[0][j] = dp[0][j-2]

        # 状态更新
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s[i-1] == p[j-1] or p[j-1] == '.':
                    dp[i][j] = dp[i-1][j-1]
                elif p[j-1] == '*':     # 【题目保证'*'号不会是第一个字符，所以此处有j>=2】
                    if s[i-1] != p[j-2] and p[j-2] != '.':
                        dp[i][j] = dp[i][j-2]
                    else:
                        dp[i][j] = dp[i][j-2] | dp[i-1][j]
        
        return dp[m][n]

#作者：flix
#链接：https://leetcode.cn/problems/regular-expression-matching/solutions/1444108/by-flix-musv/
```



##### [152. 乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/)

给你一个整数数组 `nums` ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

测试用例的答案是一个 **32-位** 整数。

**子数组** 是数组的连续子序列。

**示例 1:**

```
输入: nums = [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。
```

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if not nums: return
        res = nums[0]
        pre_max = nums[0]
        pre_min = nums[0]
        #我们只要记录前 i 的最小值，和最大值，那么 dp[i] = max(nums[i] * pre_max, nums[i] * pre_min, nums[i])，这里 0 不需要单独考虑，因为当相乘不管最大值和最小值，都会置 0
        for num in nums[1:]:
            cur_max = max(pre_max * num, pre_min * num, num)
            cur_min = min(pre_max * num, pre_min * num, num)
            res = max(res, cur_max)
            pre_max = cur_max
            pre_min = cur_min
        return res
```

##### [91. 解码方法](https://leetcode.cn/problems/decode-ways/)

一条包含字母 `A-Z` 的消息通过以下映射进行了 **编码** ：

```
'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
```

要 **解码** 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，`"11106"` 可以映射为：

- `"AAJF"` ，将消息分组为 `(1 1 10 6)`
- `"KJF"` ，将消息分组为 `(11 10 6)`

注意，消息不能分组为 `(1 11 06)` ，因为 `"06"` 不能映射为 `"F"` ，这是由于 `"6"`和 `"06"` 在映射中并不等价。

给你一个只含数字的 **非空** 字符串 `s` ，请计算并返回 **解码** 方法的 **总数** 。

题目数据保证答案肯定是一个 **32 位** 的整数。

**示例 1：**

```
输入：s = "12"
输出：2
解释：它可以解码为 "AB"（1 2）或者 "L"（12）`
```

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        f = [1] + [0] * n
        for i in range(1, n + 1):
            if s[i - 1] != '0':
                f[i] += f[i - 1]
            if i > 1 and s[i - 2] != '0' and int(s[i-2:i]) <= 26: #注意控制位数
                f[i] += f[i - 2]
        return f[n]
```



##### [309. 买卖股票的最佳时机含冷冻期](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

给定一个整数数组`prices`，其中第 `prices[i]` 表示第 `*i*` 天的股票价格 。

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

- 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。

**注意：**你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

**示例 1:**

```
输入: prices = [1,2,3,0,2]
输出: 3 
解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
```

![image-20231025222150894](/Users/kaixiwu/Library/Application Support/typora-user-images/image-20231025222150894.png)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        n = len(prices)
        #f[i][0]: 手上持有股票的最大收益
        #f[i][1]: 手上不持有股票，并且处于冷冻期中的累计最大收益
        #f[i][2]: 手上不持有股票，并且不在冷冻期中的累计最大收益
        f = [[-prices[0], 0, 0]] + [[0]*3 for _ in range(n-1)]
        for i in range(1, n):
            f[i][0] = max(f[i-1][0], f[i-1][2]-prices[i])
            f[i][1] = f[i-1][0] + prices[i]
            f[i][2] = max(f[i-1][1], f[i-1][2])
        return max(f[n-1][1], f[n-1][2])


        # 当前不持股票且不被冻结的最大收益
        # 当前被冻结的最大收益
        # 当前持有一支股票的最大收益
        #dp0, dp1, dp2 = 0, -sys.maxsize, -prices[0]           
        #for i in range(1, len(prices)):
            # 当前不持股票且不被冻结的最大收益，要么保持之前的状态，要么之前被冻结的状态现在解冻了
            # 被冻结只有持有股票卖出时会被冻结
            # 当前持有一支股票的最大收益，要么保持之前的状态，要么在不被冻结的状态下买入一支股票
        #    dp0, dp1, dp2 = max(dp0, dp1), dp2 + prices[i], max(dp2, dp0 - prices[i])   
        #return max(dp0, dp1, dp2)   # 取三者最大值

```

##### [279. 完全平方数](https://leetcode.cn/problems/perfect-squares/)

给你一个整数 `n` ，返回 *和为 `n` 的完全平方数的最少数量* 。

**完全平方数** 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，`1`、`4`、`9` 和 `16` 都是完全平方数，而 `3` 和 `11` 不是。 

**示例 1：**

```
输入：n = 12
输出：3 
解释：12 = 4 + 4 + 4
```

**示例 2：**

```
输入：n = 13
输出：2
解释：13 = 4 + 9
```



![image-20231026231049394](/Users/kaixiwu/Library/Application Support/typora-user-images/image-20231026231049394.png)

https://leetcode.cn/problems/perfect-squares/solutions/1413224/by-flix-sve5/

```python
class Solution:
    def numSquares(self, n: int) -> int:
         # 预处理出 <=sqrt(n) 的完全平方数
        nums = []
        i = 1
        while i*i <= n:
            nums.append(i*i)
            i += 1

        # 转化为完全背包问题【套用「322. 零钱兑换」】
        target = n
        dp = [target] * (target+1)    # 初始化为一个较大的值，如 +inf 或 n
        dp[0] = 0    # 合法的初始化；其他 dp[j]均不合法
        
        # 完全背包：优化后的状态转移
        for num in nums:                        # 第一层循环：遍历nums
            for j in range(num, target+1):      # 第二层循环：遍历背包【正序】
                dp[j] = min( dp[j], dp[j-num] + 1 )
        
        return dp[target]
```



##### [139. 单词拆分](https://leetcode.cn/problems/word-break/)

给你一个字符串 `s` 和一个字符串列表 `wordDict` 作为字典。请你判断是否可以利用字典中出现的单词拼接出 `s` 。

**注意：**不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

**示例 1：**

```
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。
```

**示例 2：**

```
输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以由 "apple" "pen" "apple" 拼接成。
     注意，你可以重复使用字典中的单词。
```

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        for i in range(n):
            for j in range(i+1, n+1):
                if dp[i] and (s[i:j] in wordDict):
                    dp[j] = True
        return dp[-1]


'''
作者：墙头马上
链接：https://leetcode.cn/problems/word-break/solutions/1492673/bei-bao-by-wo-zhao-wo-de-bao-zhen-aog8/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
'''
```

##### [312. 戳气球](https://leetcode.cn/problems/burst-balloons/)

有 `n` 个气球，编号为`0` 到 `n - 1`，每个气球上都标有一个数字，这些数字存在数组 `nums` 中。

现在要求你戳破所有的气球。戳破第 `i` 个气球，你可以获得 `nums[i - 1] * nums[i] * nums[i + 1]` 枚硬币。 这里的 `i - 1` 和 `i + 1` 代表和 `i` 相邻的两个气球的序号。如果 `i - 1`或 `i + 1` 超出了数组的边界，那么就当它是一个数字为 `1`的气球。

求所能获得硬币的最大数量。 

**示例 1：**

```
输入：nums = [3,1,5,8]
输出：167
解释：
nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167
```

```python
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[0] * (n + 2) for _ in range(n + 2)]
        val = [1] + nums + [1]

        for i in range(2, len(val)):
            for j in range(0, len(val) - i):
                for k in range(j+1, i+j):
                    tmp = dp[j][k] + val[j]*val[k]*val[j+i] + dp[k][i+j]
                    dp[j][i+j] = max(tmp, dp[j][i+j])
        
        return dp[0][n+1]
```

```markdown
解题思路：
假设这个区间是个开区间，最左边索引 i，最右边索引 j 我这里说 “开区间” 的意思是，我们只能戳爆 i 和 j 之间的气球，i 和 j 不要戳   DP思路是这样的，就先别管前面是怎么戳的，你只要管这个区间最后一个被戳破的是哪个气球 这最后一个被戳爆的气球就是 k   注意！！！！！ k是这个区间   最后一个   被戳爆的气球！！！！！ k是这个区间   最后一个   被戳爆的气球！！！！！   假设最后一个被戳爆的气球是粉色的，k 就是粉色气球的索引   然后由于 k 是最后一个被戳爆的，所以它被戳爆之前的场景是什么亚子？  

是这样子的朋友们！因为是最后一个被戳爆的，所以它周边没有球了！没有球了！只有这个开区间首尾的 i 和 j 了！！ 这就是为什么DP的状态转移方程是只和 i 和 j 位置的数字有关   假设 dp[i][j] 表示开区间 (i,j) 内你能拿到的最多金币   那么这个情况下   你在 (i,j) 开区间得到的金币可以由 dp[i][k] 和 dp[k][j] 进行转移   如果你此刻选择戳爆气球 k，那么你得到的金币数量就是：   total = dp[i][k] + val[i] * val[k] * val[j] + dp[k][j]   注：val[i] 表示 i 位置气球的数字 然后 (i,k) 和 (k,j) 也都是开区间   那你可能又想问了，戳爆粉色气球我能获得 val[i]*val[k]*val[j] 这么多金币我能理解(因为戳爆 k 的时候只剩下这三个气球了)， 但为什么前后只要加上 dp[i][k] 和 dp[k][j] 的值就行了呀？   因为 k 是最后一个被戳爆的，所以 (i,j) 区间中 k 两边的东西必然是先各自被戳爆了的， 左右两边互不干扰，大家可以细品一下 这就是为什么我们 DP 时要看 “最后一个被戳爆的” 气球，这就是为了让左右两边互不干扰，这大概就是一种分治的思想叭   所以你把 (i,k) 开区间所有气球戳爆，然后把戳爆这些气球的所有金币都收入囊中，金币数量记录在 dp[i][k] 同理，(k,j) 开区间你也已经都戳爆了，钱也拿了，记录在 dp[k][j] 所以你把这些之前已经拿到的钱 dp[i][k]+dp[k][j] 收着， 再加上新赚的钱 val[i]*val[k]*val[j] 不就得到你现在戳爆气球 k 一共手上能拿多少钱了吗   而你在 (i,j) 开区间可以选的 k 是有多个的，见一开头的图，除了粉色之外，你还可以戳绿色和红色 所以你枚举一下这几个 k，从中选择使得 total 值最大的即可用来更新 dp[i][j]   然后呢，你就从 (i,j) 开区间只有三个数字的时候开始计算，储存每个小区间可以得到金币的最大值 然后慢慢扩展到更大的区间，利用小区间里已经算好的数字来算更大的区间 就可以啦！撒花✿✿ヽ(^▽^)ノ✿！    


```

##### [1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/)

给定两个字符串 `text1` 和 `text2`，返回这两个字符串的最长 **公共子序列** 的长度。如果不存在 **公共子序列** ，返回 `0` 。

一个字符串的 **子序列** 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

- 例如，`"ace"` 是 `"abcde"` 的子序列，但 `"aec"` 不是 `"abcde"` 的子序列。

两个字符串的 **公共子序列** 是这两个字符串所共同拥有的子序列。 

**示例 1：**

```
输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace" ，它的长度为 3 。
```

**示例 2：**

```
输入：text1 = "abc", text2 = "abc"
输出：3
解释：最长公共子序列是 "abc" ，它的长度为 3 。
```

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        M, N =len(text1), len(text2)
        dp = [[0] * (N+1) for _ in range(M+1)]
        for i in range(1, M+1):
            for j in range(1, N+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[M][N]
```

##### [416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/)

给你一个 **只包含正整数** 的 **非空** 数组 `nums` 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

**示例 1：**

```
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。
```

**示例 2：**

```
输入：nums = [1,2,3,5]
输出：false
解释：数组不能分割成两个元素和相等的子集。
```

背包问题

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        """递推"""
        # dp = [[False for _ in range(m+1)] for _ in range(n+1)]
        # dp[0][0] = True
        # for i in range(n):
        #     for c in range(m+1):
        #         if nums[i] <= c :    
        #             dp[i+1][c] = dp[i][c-nums[i]] or dp[i][c]
        #         else:
        #             dp[i+1][c] = dp[i][c]
        # return dp[n][m]
        n = len(nums)
        m = sum(nums)
        if m % 2 == 0:
            m = m // 2
        else:
            return False

        """单一数组"""
        dp = [False for _ in range(m+1)]
        dp[0] = True
        for i in range(n):
            for c in range(m, -1, -1):
                if nums[i] <= c:
                    dp[c] = dp[c-nums[i]] or dp[c]
        return dp[m]
```

