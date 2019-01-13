# This document is used to record all study notes in learning Python

# =============================================================================
# input("") without a variable means only a blank line in output
# for and while loop can have else
# do not compare 2 floating num (because of the precision issue)
# "global" declaration is dangerous, use global variable name is dangerous
# input of a function is passed by "value", but array and list passed by init addr of them, so attay is passed by reference
# =============================================================================

difference between "==" and "is" operator:
A的車叫"小P"，和隔壁B家的車("小王")型號是一樣的。我們可以說"小P"和"小王"是一模一樣的、相等的(euqal, "==")，但本質上是兩個不同的objects。有一天A給他的車取一個別名叫"愛駒"，當我們說"小P"的時候其實就是在說"愛駒"，因為本質上兩個名字指向同一個object，這裏我們把"小P"和"愛駒"稱為完全相等的(identical , "is")。

"==": 比較兩個objects的"value"是否相等
"is": 比較兩個objects的memory addresses是否一樣

Ex:
a = [1, 2, 3]
b = a
=> a == b: True; a is b: True

c = [1, 2, 3]
=> a == c: True; a is c: False
