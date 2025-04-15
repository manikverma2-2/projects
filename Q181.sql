## 180. Consecutive Numbers

## Question
Write a SQL query to find all numbers that appear at least three times consecutively.

```
+----+-----+
| Id | Num |
+----+-----+
| 1  |  1  |
| 2  |  1  |
| 3  |  1  |
| 4  |  2  |
| 5  |  1  |
| 6  |  2  |
| 7  |  2  |
+----+-----+

For example, given the above Logs table, 1 is the only number that appears consecutively for at least three times.

+-----------------+
| ConsecutiveNums |
+-----------------+
| 1               |
+-----------------+
```

# Write your MySQL query statement below
Select distinct l1.num as ConsecutiveNums from Logs l1, Logs l2, Logs l3 where
l2.id=l1.id-1 and l3.id=l1.id-2 and l1.num=l2.num and l1.num= l3.num;

