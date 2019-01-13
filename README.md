# This document is used to record my study notes in learning Python
# =============================================================================================================

1. input("") 沒有指定input或variable: means only a blank line in output
2. "for" & "while" loop can have "else": 在for/while loop結束後跑else的內容
3. do not compare 2 floating num (because of the precision issue): 別做"float1 == float2"這種比較
4. "global" declaration is dangerous, use global variable name is dangerous
5. input of a function is passed by "value", but array and list passed by init address (passed by reference)
6. 可以用matrix運算就不要用loop, 可以用loop就不要用recursion (speed: array運算 > loop > recursion)

7. Difference between "==" and "is" operator:
"==": 比較兩個objects的"value"是否相等
"is": 比較兩個objects的memory addresses是否一樣
Ex:
a = [1, 2, 3]
b = a
=> a == b: True; a is b: True
c = [1, 2, 3]
=> a == c: True; a is c: False

