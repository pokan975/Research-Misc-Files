# This document is used for my study notes in learning Python
# =============================================================================================================

1. input("") 沒有指定input或variable: means only a blank line in output
2. "for" & "while" loop can have "else": 在for/while loop結束後跑else的內容
3. do not compare 2 floating num (because of the precision issue): 別做"float1 == float2"這種比較
4. "global" declaration is dangerous, use global variable name is dangerous
5. input of a function is passed by "value", but array and list passed by init address (passed by reference)
6. 可以用matrix運算就不要用loop, 可以用loop就不要用recursion (speed: array運算 > loop > recursion)

7. Difference between "==" and "is" operator:
"==": 比較兩個objects的"value"是否相等
"is": 比較兩個objects的memory addresses (ID)是否一樣
(DO NOT USE "IS" TO COMPARE INTEGERS!!!)
Ex:
a = [1, 2, 3]
b = a
=> a == b: True; a is b: True
c = [1, 2, 3]
=> a == c: True; a is c: False

8. Attention on No.7:
a, b = 256, 256
a is b => True           # This is an expected result
a, b = 257, 257
a is b => False          # Why is this False?
257 is 257 => True       # Yet the literal numbers compare properly
The current implementation keeps an array of integer objects for all integers between -5 and 256,
when you create an int in that range you actually just get back a reference to the existing object.
So it should be possible to change the value of 1.

9. elements in numpy.array should be number (float is better)
10. numpy and scipy both have fft function
11. when calling set() to a list, the elements will be automatically placed in order.
