#Insertion Sort - Part 2

def insertionSort(ar):
    for i in range(1, len(ar)):
        temp = ar[i]
        j = i
        while j > 0 and temp < ar[j-1]:
            ar[j] = ar[j-1]
            j -= 1
        ar[j] = temp
        print(' '.join(str(j) for j in ar))

m = input()
ar = [int(i) for i in (input().strip().split())]
insertionSort(ar)

#Linear Algebra

import numpy as np
A = np.array([input().split() for _ in range(int(input()))], float)
print(round(np.linalg.det(A),2))

#Polynomials

mport numpy as np
Coef = input().split()
X = float(input())
coef = list(map(float,Coef))
print(np.polyval(coef,X))

#Inner and Outer

import numpy as np

A = np.array(input().split(), int)
B = np.array(input().split(), int)
print(np.inner(A, B), np.outer(A, B), sep='\n')

#Dot and Cross
import numpy

n = int(input())
a = numpy.array([input().split() for _ in range(n)], int)
b = numpy.array([input().split() for _ in range(n)], int)
print(numpy.dot(a, b))

#Mean, Var, and Std
import numpy

n, m = map(int, input().split())
a = []
for i in range(n):
    a += map(int, input().split())
    
a = numpy.array(a)
a = numpy.reshape(a, (n,m))

print(numpy.mean(a, axis = 1))
print(numpy.var(a, axis = 0))
print(numpy.round(numpy.std(a), 11))

#Min and Max

import numpy

N, M = map(int, input().split())
A = numpy.array([input().split() for _ in range(N)], int)
print(numpy.max(numpy.min(A, axis=1), axis=0))

#Sum and Prod
import numpy
N, M = tuple(map(int, input().split()))
running_list = []
for i in range(0, N):
    running_list.append(input().split())
    
my_array = numpy.array(running_list, int)
print(numpy.prod(numpy.sum(my_array, axis=0), axis=None))

#Floor, Ceil and Rint
import numpy
numpy.set_printoptions(sign=' ')
A = numpy.array(input().split(), float)
print(numpy.floor(A))
print(numpy.ceil(A))
print(numpy.rint(A))

#Array Mathematics

# Enter your code here. Read input from STDIN. Print output to STDOUT

import numpy as np

n, m = map(int, input().split())
a, b = (np.array([input().split() for _ in range(n)], dtype=int) for _ in range(2))
print(a + b, a - b, a * b, a // b, a % b, a ** b, sep='\n')

#Eye and Identity

# Enter your code here. Read input from STDIN. Print output to STDOUT

import numpy

numpy.set_printoptions(sign=' ')
print(numpy.eye(*map(int, input().split())))

#Zeros and Ones

import numpy
N = tuple(map(int, input().split()))
print(numpy.zeros(N, int))
print(numpy.ones(N, int))

#Concatenate

# Enter your code here. Read input from STDIN. Print output to STDOUT

import numpy as np

N, M, P = map(int, input().split())

lt = []
for i in range(N):
    N_ = input().split()
    n = list(map(int,N_))
    lt.append(n)
mt = []
for i in range(M):
    M_ = input().split()
    m = list(map(int,M_))
    mt.append(m)

result = np.concatenate((lt,mt)).reshape(N+M,P)
print(result)

#Transpose and Flatten

import numpy

N, M = map(int, input().split())

lst = []

for i in range(0, N):
    lst.append(list(map(int, input().split())))
    
arr = numpy.array(lst)

print(numpy.transpose(arr))
print(arr.flatten())

#Shape and Reshape

import numpy

arr = numpy.array(input().split(), int)

print(numpy.reshape(arr,(3,3)))

#Arrays

def arrays(arr):
    # complete this function
    arr = arr[::-1]
    return numpy.array(arr, float)

#Decorators 2 - Name Directory

def person_lister(f):
    def inner(people):
        people = sorted(people, key=lambda person: int(person[2]))
        return [f(person) for person in people]
    return inner

#Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        # complete the function
        # now I got to 10
        result = []
        
        for _ in l:
            if len(_) != 10:
                if _.startswith('0'):
                    _ = _[1:]
                elif _.startswith('+91'):
                    _ = _[3:]
                elif _.startswith('91'):
                    _ = _[2:]
            if len(_) == 10:
                result.append('+91 ' + _[:5] + ' ' + _[5:10] )

        print(*sorted(result), sep='\n')
        
            
    return fun

#Arithmetic Operators

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
print(a + b)
print(a - b)
print(a * b)

#Python If-Else

#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())

if n % 2 != 0:
    print('Weird')
elif 2 < n <= 5:
    print('Not Weird')
elif 6 <= n <= 20:
    print('Weird')
elif n > 20:
    print('Not Weird')

#Say "Hello, World!" With Python

if __name__ == '__main__':
    print("Hello, World!")

#Find Angle MBC

import math

x = int(input())
y = int(input())

print(math.ceil(180 - (90 * y/x + 45 * x/y)), chr(176), sep="")


#Between Two Sets

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'getTotalX' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER_ARRAY a
#  2. INTEGER_ARRAY b
#

def getTotalX(a, b):
    # Write your code here
    results = []
    for i in range(min(a), (max(b)+1)):
        check = False
        for j in range(0, n):
            if i % a[j] != 0:
                check = True
                continue
            if check:
                continue
        for p in range(0, m):
            if b[p] % i != 0:
                check = True
                continue
            if check:
                continue
        if not check:
            results.append(i)
    return len(results)
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = int(first_multiple_input[0])

    m = int(first_multiple_input[1])

    arr = list(map(int, input().rstrip().split()))

    brr = list(map(int, input().rstrip().split()))

    total = getTotalX(arr, brr)

    fptr.write(str(total) + '\n')

    fptr.close()

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'getTotalX' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER_ARRAY a
#  2. INTEGER_ARRAY b
#

def getTotalX(a, b):
    # Write your code here
    results = []
    for i in range(min(a), max(b)):
        check = False
        for j in range(0, n):
            if i % a[j] != 0:
                check = True
                continue
            if check:
                continue
        for p in range(0, m):
            if b[p] % i != 0:
                check = True
                continue
            if check:
                continue
        if not check:
            results.append(i)
    return len(results)
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = int(first_multiple_input[0])

    m = int(first_multiple_input[1])

    arr = list(map(int, input().rstrip().split()))

    brr = list(map(int, input().rstrip().split()))

    total = getTotalX(arr, brr)

    fptr.write(str(total) + '\n')

    fptr.close()

#Apple and Orange

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'countApplesAndOranges' function below.
#
# The function accepts following parameters:
#  1. INTEGER s
#  2. INTEGER t
#  3. INTEGER a
#  4. INTEGER b
#  5. INTEGER_ARRAY apples
#  6. INTEGER_ARRAY oranges
#

def countApplesAndOranges(s, t, a, b, apples, oranges):
    # Write your code here
    app = 0
    ora = 0
    for i in range(0, m):
        if s <= (a + apples[i]) <= t:
            app += 1
    for i in range(0, n):
        if s <= (b + oranges[i]) <= t:
            ora += 1
    print(app)
    print(ora)
    return

if __name__ == '__main__':
    first_multiple_input = input().rstrip().split()

    s = int(first_multiple_input[0])

    t = int(first_multiple_input[1])

    second_multiple_input = input().rstrip().split()

    a = int(second_multiple_input[0])

    b = int(second_multiple_input[1])

    third_multiple_input = input().rstrip().split()

    m = int(third_multiple_input[0])

    n = int(third_multiple_input[1])

    apples = list(map(int, input().rstrip().split()))

    oranges = list(map(int, input().rstrip().split()))

    countApplesAndOranges(s, t, a, b, apples, oranges)

#Grading Students

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'gradingStudents' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts INTEGER_ARRAY grades as parameter.
#

def gradingStudents(grades):
    results = []
    for _ in grades:
        if _ >= 38 and (_ % 5) >= 3:
            _ += 5 - (_ % 5)
        results.append(_)
    return (results)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    grades_count = int(input().strip())

    grades = []

    for _ in range(grades_count):
        grades_item = int(input().strip())
        grades.append(grades_item)

    result = gradingStudents(grades)

    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')

    fptr.close()

#Insertion Sort - Part 1

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort1(n, arr):
    tmp = arr[-1]
    for i in range(n-2, -1, -1):
        if arr[i] > tmp:
            arr[i+1] = arr[i]
            print(' '.join(map(str, arr)))
        else:
            arr[i+1] = tmp
            print(' '.join(map(str, arr)))
            return

    arr[0] = tmp
    print(' '.join(map(str, arr)))

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

#Time Conversion

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'timeConversion' function below.
#
# The function is expected to return a STRING.
# The function accepts STRING s as parameter.
#

def timeConversion(s):
    # Write your code here
    time = s[:-2]
    h, m, sec = map(int, time.split(":"))
    
    if s[-2:] == "PM" and h != 12:
        h += 12
    elif s[-2:] == "AM" and h == 12:
        h = 0
        
    return f"{h:02}:{m:02}:{sec:02}"

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = timeConversion(s)

    fptr.write(result + '\n')

    fptr.close()

#Mini-Max Sum

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'miniMaxSum' function below.
#
# The function accepts INTEGER_ARRAY arr as parameter.
#

def miniMaxSum(arr):
    # Write your code here
    ar = list(arr)
    arr.remove(min(arr))
    ar.remove(max(ar))
    print(sum(ar), sum(arr))
    return

if __name__ == '__main__':

    arr = list(map(int, input().rstrip().split()))

    miniMaxSum(arr)

#Staircase

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'staircase' function below.
#
# The function accepts INTEGER n as parameter.
#

def staircase(n):
    # Write your code here
    for i in range(0, n):
        row = ''.join((n-i-1) * [' '] + (i + 1) * ['#'])
        print(row)
    return

if __name__ == '__main__':
    n = int(input().strip())

    staircase(n)

#Plus Minus

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'plusMinus' function below.
#
# The function accepts INTEGER_ARRAY arr as parameter.
#

def plusMinus(arr):
    # Write your code here
    p = 0
    n = 0
    z = 0
    for i in range(0, len(arr)):
        if arr[i] > 0:
            p += 1
        elif arr[i] < 0:
            n += 1
        elif arr[i] == 0:
            z += 1
    print('%.6f' % (p / len(arr)))
    print('%.6f' % (n / len(arr)))
    print('%.6f' % (z / len(arr)))
    return
    
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    plusMinus(arr)

#Diagonal Difference

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'diagonalDifference' function below.
#
# The function is expected to return an INTEGER.
# The function accepts 2D_INTEGER_ARRAY arr as parameter.
#

def diagonalDifference(arr):
    # Write your code here
    p = 0
    s = 0
    for i in range(0, len(arr)):
        p += arr[i][i]
        s += arr[i][(len(arr) - 1) - i]
    res = abs(p - s)
    return res

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    result = diagonalDifference(arr)

    fptr.write(str(result) + '\n')

    fptr.close()

#A Very Big Sum

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'aVeryBigSum' function below.
#
# The function is expected to return a LONG_INTEGER.
# The function accepts LONG_INTEGER_ARRAY ar as parameter.
#

def aVeryBigSum(ar):
    # Write your code here
    return (sum(ar))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input().strip())

    ar = list(map(int, input().rstrip().split()))

    result = aVeryBigSum(ar)

    fptr.write(str(result) + '\n')

    fptr.close()

#Compare the Triplets
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'compareTriplets' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts following parameters:
#  1. INTEGER_ARRAY a
#  2. INTEGER_ARRAY b
#

def compareTriplets(a, b):
    # Write your code here
    sa = 0
    sb = 0
    for i in range(0, len(a)):
        if a[i] > b[i]:
            sa += 1
        elif a[i] < b[i]:
            sb += 1
    
    return (sa, sb)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    a = list(map(int, input().rstrip().split()))

    b = list(map(int, input().rstrip().split()))

    result = compareTriplets(a, b)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()

#Simple Array Sum

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'simpleArraySum' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY ar as parameter.
#

def simpleArraySum(ar):
    # Write your code here
    return (sum(ar))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input().strip())

    ar = list(map(int, input().rstrip().split()))

    result = simpleArraySum(ar)

    fptr.write(str(result) + '\n')

    fptr.close()

#Solve Me First

def solveMeFirst(a,b):
	# Hint: Type return a+b below
    return (a+b)

num1 = int(input())
num2 = int(input())
res = solveMeFirst(num1,num2)
print(res)

#Recursive Digit Sum

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#

def superDigit(n, k):
    return 1 + (k * sum(int(x) for x in n) - 1) % 9

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

#Viral Advertising

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    
    Shared = 5
    Liked = 2
    Cumulative = 2
    
    for i in range(1, n):
        Shared = Liked * 3
        Liked = math.floor(Shared/2)
        Cumulative += Liked
    
    return Cumulative
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

#Number Line Jumps

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1 0 3 6 9 12
#  2. INTEGER v1 3 
#  3. INTEGER x2 4 6 8 10 12
#  4. INTEGER v2 2
#


def kangaroo(x1, v1, x2, v2):
    
    if v1 <= v2:
        return "NO"
        
    distance_to_cover = x2 - x1
    
    if distance_to_cover % (v2 - v1) == 0:
        return "YES"
    else:
        return "NO"

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

#Birthday Cake Candles

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    # Write your code here
    tallest = max(candles)
    return candles.count(tallest)
    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

#XML2 - Find the Maximum Depth

maxdepth = -1
def depth(elem, level):
    global maxdepth
    # your code goes here
    if level == (maxdepth):
        maxdepth += 1
    for child in elem:
        depth(child, level + 1)
 
#XML 1 - Find the Score

def get_attr_number(node):
    # your code goes here
    Score = 0
    for element in root.iter():
        Score += len(element.attrib)
    return(Score)

#Matrix Script

#!/bin/python3

import math
import os
import random
import re
import sys




first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []
string = ''

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)
    
for i in range(0, m):
    for j in range(0, n):
        string += matrix[j][i]
 
print(re.sub(r'(?<=\w)([^\w\d]+)(?=\w)', ' ', string))
            

#Validating Postal Codes

regex_integer_in_range = r'^[1-9][\d]{5}$'
regex_alternating_repetitive_digit_pair = r'(\d)(?=\d\1)'

#HTML Parser - Part 1

# Enter your code here. Read input from STDIN. Print output to STDOUT

from html.parser import HTMLParser

N = int(input())

class MyHTMLParser(HTMLParser):
    
        def handle_starttag(self, tag, attrs):
            print('Start :', tag)
            for element in attrs:
                print('->', element[0], '>', element[1])
                
        def handle_endtag(self, tag):
            print('End   :', tag)
        
        def handle_startendtag(self, tag, attrs):
            print('Empty :', tag)
            for element in attrs:
                print('->', element[0], '>', element[1])
                
Parser = MyHTMLParser()
Parser.feed(''.join([input().strip() for i in range(0, N)]))

#Hex Color Code

# Enter your code here. Read input from STDIN. Print output to STDOUT

import re

N = int(input())
pattern = r"(?<!^)#[0-9a-fA-F]{3,6}"

for i in range(0, N):
    Line = input()
    matches = re.findall(pattern, Line)
    if matches:
        for _ in matches:
            print(_)

#Validating and Parsing Email Addresses

# Enter your code here. Read input from STDIN. Print output to STDOUT

import re
import email.utils

"""
n = int(input())
email_pattern = re.compile(r'^[a-z][\w\-\.]+@[a-z]+\.[a-z]{1,3}')

for i in range(0, n):
    parsed_addr = email.utils.parseaddr(input())
    if re.search(email_pattern, parsed_addr[1]):
        print(email.utils.formataddr(parsed_addr))
"""
n = int(input())
for _ in range(n):
    x, y = input().split(' ')
    m = re.match(r'<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>', y)
    if m:
        print(x,y)

#Validating phone numbers

# Enter your code here. Read input from STDIN. Print output to STDOUT

import re

N = int(input())
pattern = re.compile(r'^[789]\d{9}$')

for i in range(0, N):
    if pattern.match(input()):
        print("YES")
    else:
        print("NO")

#Validating Roman Numerals

regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"

#Regex Substitution

import re

ii = int(input())

for i in range(0,ii):
   txt = input()
   txt = re.sub(r"\ \&\&\ "," and ",txt)
   txt = re.sub(r"\ \|\|\ "," or ",txt)
   txt = re.sub(r"\ \&\&\ "," and ",txt)
   txt = re.sub(r"\ \|\|\ "," or ",txt)
   print(txt)

#Re.start() & Re.end()

# Enter your code here. Read input from STDIN. Print output to STDOUT

import re

string = input()
substring = input()

pattern = re.compile(substring)
match = pattern.search(string)
if not match:
    print('(-1, -1)')
while match:
    print('({0}, {1})'.format(match.start(), match.end() - 1))
    match = pattern.search(string, match.start() + 1)

#Re.findall() & Re.finditer()

# Enter your code here. Read input from STDIN. Print output to STDOUT

import re

vowels = 'aeiou'
consonants = 'qwrtypsdfghjklzxcvbnm'
match = re.findall(r'(?<=[' + consonants + '])([' + vowels + ']{2,})(?=[' + consonants + '])', input(), flags=re.I)
print('\n'.join(match or ['-1']))

#Group(), Groups() & Groupdict()

# Enter your code here. Read input from STDIN. Print output to STDOUT

import re

s = input()
m = re.search(r'([a-zA-Z0-9])\1+', s)

if m:
    print(m.group(1))
else:
    print(-1)

#Group(), Groups() & Groupdict()

# Enter your code here. Read input from STDIN. Print output to STDOUT

import re

s = input()
m = re.search(r'([a-zA-Z0-9])\1+', s)

if m:
    print(m.group(1))
else:
    print(-1)

#Re.split()

regex_pattern = r'[.,]'

#Detect Floating Point Number

# Enter your code here. Read input from STDIN. Print output to STDOUT

import re

T = int(input())  # Number of test cases

def isfloat(s):
    if not bool(re.match(r'^[0-9\+\-\.]+$', s)):
        return (False)
    if not bool(re.match(r'^[^.]*\.[^.]*$', s)):
        return (False)
    if s.endswith('.'):
        return (False)
    if not bool(re.match(r'^[^\+\-]*[\+\-]?[^\+\-]*$', s)):
        return (False)
    if not (s.startswith('+') or s.startswith('-')):
        return (False)
    return (True)

for i in range(0, T):  # Getting the test cases and checking the result
    s = input()
   # print(isfloat(s))
    print(bool(re.match(r'^[-+]?[0-9]*\.[0-9]+$', s)))
    """
        the conditions:
            1 - check if is only digit - + .
            2 - has only one . , not in the end
            3 - has maximum one - or + , just in the begining
                 
    """
    

#Map and Lambda Function


cube = lambda x: x ** 3

def fibonacci(n):
    fibs = [0, 1]
    if n > 2:
        for i in range(2, n):
            fibs.append(int(fibs[i-1] + fibs[i-2]))
    return (fibs[0:n])

#ginortS

# Enter your code here. Read input from STDIN. Print output to STDOUT

s = list(input())

up = []
low = []
odd = []
eve =[]

for _ in s:
    if _.isupper():
        up.append(_)
    elif _.islower():
        low.append(_)
    elif _.isdigit():
        if int(_) % 2 == 0:
            eve.append(_)
        else:
            odd.append(_)
        
up.sort()
low.sort()
odd.sort()
eve.sort()

for _ in low:
    print(_, end="")
for _ in up:
    print(_, end="")
for _ in odd:
    print(_, end="")
for _ in eve:
    print(_, end="")
    
#Athlete Sort

#!/bin/python3

import math
import os
import random
import re
import sys

atts = []

def sort_by_k(k):
    for j in range(0, (n-1)):
        for i in range(0, (n-1)):
            if arr[i][k] > arr[i+1][k]:
                temporary = arr[i]
                arr[i] = arr[i+1]
                arr[i+1] = temporary
            
if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    sort_by_k(k)
    
    for sublist in arr:
        print(*sublist)
    
#Zipped!

# Enter your code here. Read input from STDIN. Print output to STDOUT

# N students in X subjects

N, X = map(int, input().split())

IDs = [i for i in range(1, N+1)]
Records = []

for i in range(0, X):
    Records.append(list(map(float, input().split())))

Sheet = list(zip(*Records))

for column in Sheet:
    average = sum(column) / len(column)
    print("{:.2f}".format(average))

#Piling Up!

# Enter your code here. Read input from STDIN. Print output to STDOUT

# horizontal row of n cubes, with the given length

from collections import deque
"""
T = int(input())  # number of test cases

for i in range(0, T):
    pile_size = int(input())
    pile = deque(list(input().split()))
    vertical = []
    
    if pile[-1] >= pile[0]:
        vertical.append(pile[-1])
        pile.pop()
    else:
        vertical.append(pile[0])
        pile.popleft()
        
    for j in range(0, (pile_size - 1)):
        if pile[-1] >= vertical[-1]:
            vertical.append(pile[-1])
            pile.pop()
        elif pile[0] >= vertical[-1]:
            vertical.append(pile[0])
            pile.popleft()
        elif len(pile) > 0:
            print("No")
            break
        else:
            print("Yes")
            break
"""

for t in range(int(input())):
    input()
    lst = [int(i) for i in input().split()]
    min_list = lst.index(min(lst))
    left = lst[:min_list]
    right = lst[min_list+1:]
    if left == sorted(left,reverse=True) and right == sorted(right):
        print("Yes")
    else:
        print("No")


#Time Delta

#!/bin/python3

import math
import os
import random
import re
import sys
from datetime import datetime

# Complete the time_delta function below.

def time_delta(t1, t2):
    a=datetime.strptime(t1, "%a %d %b %Y %H:%M:%S %z")
    b=datetime.strptime(t2, "%a %d %b %Y %H:%M:%S %z")
    
    return str(int(abs((a-b).total_seconds())))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()


#Exceptions

# Enter your code here. Read input from STDIN. Print output to STDOUT

T = int(input())

for i in range(0, T):
    a, b = input().split()
    try:
        print(int(int(a)/int(b)))
    except ZeroDivisionError:
        print("Error Code: integer division or modulo by zero")
    except ValueError as e:
        print("Error Code:", e)

#Calendar Module

# Enter your code here. Read input from STDIN. Print output to STDOUT

import calendar
 
date = list(map(int, input().split()))

day = (calendar.weekday(date[2], date[0], date[1]))

if day == 0 : print("MONDAY")
if day == 1 : print("TUESDAY")
if day == 2 : print("WEDNESDAY")
if day == 3 : print("THURSDAY")
if day == 4 : print("FRIDAY")
if day == 5 : print("SATURDAY")
if day == 6 : print("SUNDAY")

#Company Logo

#!/bin/python3

import math
import os
import random
import re
import sys
from collections import deque, OrderedDict

'''
    three top most characters:
    - print with their occurance count
    - sort in descending occurnce count
    - if the occurance is the same, print in alphabetic order
'''


if __name__ == '__main__':
    '''
    s = input()
    characters = {}  # to store the count of characters
    for _ in s:
        characters[_] = s.count(_)
    occurances = sorted(set(characters.values()))
    result = []

    for i in range(3):
        repeat = 0
        for j in len(characters):
            if characters[j] == occurances[i]:
                result.append(characters[])
                repeat += 1
        i += repeat
        
    print(result)
    '''
    from collections import Counter

    s = input()
    for x,y in Counter(sorted(s)).most_common(3):
        print(f"{x} {y}")
    
#Collections.deque()

# Enter your code here. Read input from STDIN. Print output to STDOUT

from collections import deque

N = int(input())
d = deque()

for i in range(N):
    op = list(map(str, input().split()))
    if len(op) > 1:
        if op[0] == 'append':
            d.append(op[1])
        elif op[0] == 'appendleft':
            d.appendleft(op[1])
    else:
        if op[0] == 'popleft':
            d.popleft()
        elif op[0] == 'pop':
            d.pop()
    
for _ in d:
    print(_, end=' ')
    
#Word Order

# Enter your code here. Read input from STDIN. Print output to STDOUT

# imports
from collections import OrderedDict

# inputs and definitions
n = int(input())
Dict = OrderedDict()
for i in range(n):
    word = input()
    if (word) in Dict:
        Dict[word] += 1
    else:
        Dict[word] = 1
        
print(len(Dict))
for _ in Dict:
    print(Dict[_], end=' ')
        

#Collections.OrderedDict()

# Enter your code here. Read input from STDIN. Print output to STDOUT

'''
    manager of a super market
    list of N items with prices bought on a particular day
    print each item name and net price in order
    
'''

from collections import OrderedDict
ordinary_dict = OrderedDict()

N = int(input())

for i in range(N):
    purchase = list(map(str, input().split()))
    price = int(purchase[-1])
    purchase.pop()
    name = ' '.join(purchase)
    if name in ordinary_dict:
        ordinary_dict[name] += price
    else:
        ordinary_dict[name] = price
    
for _ in ordinary_dict:
    print(_, ordinary_dict[_])

#Collections.namedtuple()

# Enter your code here. Read input from STDIN. Print output to STDOUT

from collections import namedtuple

N = int(input())  # the total number of students
column_names = input()
record = namedtuple('record', column_names)
spreadshit = []
average = 0

for i in range(N):
    a, b, c, d = map(str, input().split())
    j = record(a, b, c, d)
    spreadshit.append(j)

for _ in spreadshit:
    average += int(_.MARKS) / len(spreadshit)

print(average)

#DefaultDict Tutorial

# Enter your code here. Read input from STDIN. Print output to STDOUT

""" 
    n words in group A , m words in group B
    for all m words in group B we check if they are exist in
    group A or not.
    if they exist in A, print the indices, if not, print -1.
    
"""
from collections import defaultdict
d = defaultdict(list)
list1=[]
n, m = map(int, input().split())
for i in range(1, n+1):
    d[input()].append(str(i))


for i in range(m):
    b = input()
    if b in d:
        print(' '.join(d[b]))
    else:
        print(-1)

#Check Strict Superset

# Enter your code here. Read input from STDIN. Print output to STDOUT

def isstrictsuperset(a,b):
    return b.issubset(a) and not(a.issubset(b))

a = set(int(x) for x in input().split(' '))
n = int(input())
res = True

for _ in range(n):
    b = set(int(x) for x in input().split(' '))
    res &= isstrictsuperset(a,b)
    
print(res)

#Check Subset

# Enter your code here. Read input from STDIN. Print output to STDOUT

N = int(input())

for i in range(0, N):
    lA = int(input())
    A = set(map(int, input().split()))
    lB = int(input())
    B = set(map(int, input().split()))
    if B.intersection(A) == A:
        print("True")
    else:
        print("False")

#The Captain's Room

# Enter your code here. Read input from STDIN. Print output to STDOUT

# must check the number of repaets of the room number

# inputs
K = int(input())
n = list(map(int, input().split()))
s = set(n)

# checked the solution from internet

print(int((sum(s)*K - sum(n))/(K-1)))

#Set Mutations

# Enter your code here. Read input from STDIN. Print output to STDOUT

# inputs
L = int(input())
A = set(map(int, input().split()))
N = int(input())

for i in range(0, N):
    op = input().split()
    if "intersection_update" in op:
        A.intersection_update(set(map(int, input().split())))
    if "difference_update" in op:
        A.difference_update(set(map(int, input().split())))
    if "symmetric_difference_update" in op:
        A.symmetric_difference_update(set(map(int, input().split())))
    if "update" in op:
        A.update(set(map(int, input().split())))
        
print(sum(A))


#Set .symmetric_difference() Operation

# Enter your code here. Read input from STDIN. Print output to STDOUT

neng = int(input())
eng = set(map(int, input().split()))
nfr = int(input())
fr = set(map(int, input().split()))

print(len(eng.symmetric_difference(fr)))

#Set .difference() Operation

# Enter your code here. Read input from STDIN. Print output to STDOUT

neng = int(input())
eng = set(map(int, input().split()))
nfr = int(input())
fr = set(map(int, input().split()))

print(len(eng.difference(fr)))

#Set .intersection() Operation

# Enter your code here. Read input from STDIN. Print output to STDOUT

negn = int(input())
eng = set(map(int, input().split()))
nfr = int(input())
fr = set(map(int, input().split()))

print(len(eng.intersection(fr)))

#Set .union() Operation

# Enter your code here. Read input from STDIN. Print output to STDOUT

# inputs
neng = int(input())
eng = set(map(int, input().split()))
nfr = int(input())
fr = set(map(int, input().split()))

print(len(eng.union(fr)))

#Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
c = int(input())

# getting and doing commands
for i in range(0, c):
    fd = input().split()
    if "remove" in fd:
        s.discard(int(fd[1]))
    elif "discard" in fd:
        s.discard(int(fd[1]))
    elif "pop" in fd:
        s.pop()
    
# pritning
print(sum(s))

#Symmetric Difference

# Enter your code here. Read input from STDIN. Print output to STDOUT

a,b = [set(input().split()) for _ in range(4)][1::2]
print('\n'.join(sorted(a^b, key=int)))

#Set .add()

# Enter your code here. Read input from STDIN. Print output to STDOUT

N = int(input())
stamps = set()
for i in range(0, N):
    stamps.add(input())
    
print(len(set(stamps)))

#No Idea!

# Enter your code here. Read input from STDIN. Print output to STDOUT

n, m = input().split()

S = input().split()

A = set(input().split())
B = set(input().split())
print(sum([(i in A) - (i in B) for i in S]))

#Merge the Tools!

def merge_the_tools(s, k):
    # your code goes here
    # string s with length n charachters
    # int k a factor of n
    n = len(s)
    r = []
    ir = int(n / k)
    for i in range (0, ir):
        current = s[(i*k):((i+1)*k)]
        for j in range (0, k):
            if current[j] not in r: r.append(current[j])
        print ("".join(r))
        r = []

#Introduction to Sets


def average(arr):
    # your code goes here
    m = sum(set(arr)) / len(set(arr))
    return(m)
        
#Polar Coordinates

# Enter your code here. Read input from STDIN. Print output to STDOUT


import cmath

print(*cmath.polar(complex(input())), sep='\n')

#collections.Counter()

# Enter your code here. Read input from STDIN. Print output to STDOUT

import collections

X = int( input() )
Shoes = collections.Counter( map( int , input().split() ) )
income = 0

N = int( input() )
for i in range ( 0 , N ):
    size, price = map(int, input().split())
    if Shoes[size]: 
        income += price
        Shoes[size] -= 1
        
print(income)

#itertools.combinations()

# Enter your code here. Read input from STDIN. Print output to STDOUT

from itertools import combinations 

S , N = input() . split()

for i in range(1, int(N)+1):
    for j in combinations(sorted(S), i):
        print(''.join(j))

#itertools.permutations()

# Enter your code here. Read input from STDIN. Print output to STDOUT


from itertools import permutations

S , n = input() . split() 
N = int(n)

I = list( permutations( S , N ))
I.sort()

for _ in I :
   for i in range ( 0 , N ):
       print ( _[i] , end='' )
   print ( '' )

#itertools.product()

# Enter your code here. Read input from STDIN. Print output to STDOUT

from itertools import product

A = map( int , input().split() )
B = map( int , input().split() )
P = list( product ( A , B ) )

for _ in P : print ( _ , end=' ' )

#The Minion Game

def minion_game(s):
    # your code goes here
    n = len(s)
    comb = ((n)*(n+1))/2
    count_k = 0
    count_s = 0
    count_k = sum([len(s[i:]) for i in range(len(s)) if s[i] in "AEIOU"])
    count_s = comb - count_k
    
    if count_s == count_k:
        print("Draw")
    elif count_s > count_k:
        print("Stuart", int(count_s) )
    else:
        print("Kevin", int(count_k))
            
#Capitalize!



# Complete the solve function below.
def solve(s):
    for x in s[:].split():
        s = s.replace(x, x.capitalize())
    return s

#Alphabet Rangoli

def print_rangoli(size):
    # your code goes here\
    import string
    design = string.ascii_lowercase
    L = []
    for i in range(n):
        s = "-".join(design[i:n])
        L.append((s[::-1]+s[1:]).center(4*n-3, "-"))
        
    print('\n'.join(L[:0:-1]+L))

#String Formatting

def print_formatted(number):
    # your code goes here
    
    l1 = len(bin(number)[2:])
   
    for i in range(1,number+1):
        print(str(i).rjust(l1,' '),end=" ")
        print(oct(i)[2:].rjust(l1,' '),end=" ")
        print(((hex(i)[2:]).upper()).rjust(l1,' '),end=" ")
        print(bin(i)[2:].rjust(l1,' '),end=" ")
        print("")
 
#Designer Door Mat

# Enter your code here. Read input from STDIN. Print output to STDOUT

n, m = map(int,input().split())
for i in range(n//2):
    j = int((2*i)+1)
    print(('.|.'*j).center(m, '-'))
print('WELCOME'.center(m,'-'))
for i in reversed(range(n//2)):
    j = int((2*i)+1)
    print(('.|.'*j).center(m, '-'))

#Text Wrap

def wrap(string, max_width):
    result = ''
    j = 0
    for i in range ( 0 , len(string)):
        j += 1 
        result += string[i]
        if j % max_width == 0 : result += '\n'
    return result

#Text Alignment

# Enter your code here. Read input from STDIN. Print output to STDOUT
thickness = int(input())
c = 'H'


for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))


for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))


for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    


for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    


for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


#String Validators

if __name__ == '__main__':
    s = input()
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    for i in range ( 0 , len(s) ):
        if s[i].isalnum() == True : a = 1
        if s[i].isalpha() == True : b = 1
        if s[i].isdigit() == True : c = 1
        if s[i].islower() == True : d = 1
        if s[i].isupper() == True : e = 1
    
    if a == 1:
        print ('True')
    else:
        print ('False')
    if b == 1: 
        print ('True')
    else:
        print ('False')
    if c == 1:
        print ('True')
    else:
        print ('False')
    if d == 1:
        print ('True')
    else:
        print ('False')
    if e == 1:
        print ('True')
    else:
        print ('False')

#Find a string

def count_substring(string, sub_string):
    l = len(string)
    x = 0
    for i in range ( 0 , l ):
        if string[i:].startswith(sub_string):
            x += 1
    return x

#Mutations

def mutate_string(s, i, c):
    l = list(s)
    l[i] = c
    s = ''.join(l)
    return (s)

#What's Your Name?

def print_full_name(first_name, last_name):
    # Write your code here
    return(print('Hello' , first_name , last_name + '! You just delved into python.' ))

#String Split and Join

def split_and_join(line):
    # write your code here
    line = line.split(" ")
    line = "-".join(line)
    return(line)

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

#sWAP cASE

def swap_case(s):
    return (s.swapcase())

#Tuples

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t = tuple(integer_list)
    print(hash(t))

#Lists

if __name__ == '__main__':
    N = int(input())

arry = []

for i in range ( 0 , N ):
    cmnd = input()
    if cmnd.find('insert') >= 0 :
        x = cmnd.strip('insert ')
        postn , numb = x.split()
        arry.insert( int(postn), int(numb) )
    if cmnd.find('print') >= 0 :
        print(arry)
    if cmnd.find('remove') >= 0 :
        x = cmnd.strip('remove ')
        arry.remove(int(x))
    if cmnd.find('append') >= 0 :
        x = cmnd.strip('append ')
        arry.append(int(x))
    if cmnd.find('sort') >= 0 :
        arry.sort()
    if cmnd.find('pop') >= 0 :
        arry.pop()
    if cmnd.find('reverse') >= 0 :  
        arry.reverse()
        
#Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
x = student_marks.get(query_name)
print( "%.2f" % (( x[0] + x [1] + x [2] ) / 3))

#Nested Lists

records = []
n = 0
if __name__ == '__main__':
    for _ in range(int(input())):
        n = n + 1
        name = input()
        score = float(input())
        records.append ( [ score , name ] ) #adds all the values in a list

records.sort() #sorts the values
lowest = second_lowest = records[0][0]
i = 0
j = 0

while i < n:
    if records[i][0] > second_lowest:
        second_lowest = records[i][0]
        j += 1
    if j == 1: print (records[i][1])
    if lowest != second_lowest and j > 1 : break
    i += 1

#Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    
runup = []
for i in arr: runup.append(i)

runup.sort( reverse = True )

for j in range ( 0 , n-1 ):
    if runup[j] > runup[j+1]: 
        print ( runup[j+1] )
        break
    
#List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
cube = []
    
for i in range ( 0 , x+1):
    for j in range ( 0 , y+1 ):
        for k in range ( 0 , z+1 ):
            if ( i + j + k != n) : cube.append( [i , j , k] ) 
            
print (cube)


#Print Function

if __name__ == '__main__':
    n = int(input())
    o = 0
    p = 1
    for i in range ( 1 , n+1 ) : 
        if i == 10 : p = p + 1
        if i == 100 : p = p + 1
        o = o * 10 ** p + i
    print (o)     

#Write a function

def is_leap(year):
    leap = False
    
    if year % 4 == 0 : leap = True
    if year % 100 == 0 : leap = False
    if year % 400 == 0 : leap = True
    
    return leap

#Loops

if __name__ == '__main__':
    n = int(input())
    
    for i in range ( 0 , n ):
        print ( i * i )

#Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())

    print ( a // b )
    print ( a / b )

#Python If-Else

#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
if n % 2 > 0:
    print ("Weird")
if n % 2 == 0 and n < 5:
    print ("Not Weird")
if n % 2 == 0 and n > 5 and n < 21:
    print ("Weird")
if n % 2 == 0 and n > 20:
    print ("Not Weird") 

#Say "Hello, World!" With Python

if __name__ == '__main__':
    print("Hello, World!")

#Arithmetic Operators

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
print (a + b)
print (a - b)
print (a * b)

