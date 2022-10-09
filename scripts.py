#TARANTINO RAMONA, ramonatarantino00@gmail.com 
#HOMEWORK_1 ----
#PROBLEM 1



#Say “Hello, World” With Python
print("Hello, World!")
#Python If-Else
import math
import os
import random
import re
import sys


if __name__ == '__main__':
    n = int(input().strip())
if n%2 == 1: 
    print('Weird')
elif n in range(2,5):
    print('Not Weird')
elif n in range(6,21): 
    print('Weird')
if n%2==0 and n>20:
    print('Not Weird')
#Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input()) 
    print(a+b)
    print(a-b)
    print(a*b)
    

#Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(int(a/b))
    print(float(a/b))
#Loops

if __name__ == '__main__':
    n = int(input())
for i in range(n): 
    print (i**2)
#Write a function
def is_leap(year):
    leap = False
    
   
    if (year%4==0 and year%100!=0):
        return True
    elif (year%400 ==0):
        return True
    else: 
        return False 
    return leap

year = int(input())
print(is_leap(year))

#Print function
if __name__ == '__main__':
    n = int(input())
for i in range(n): 
    print(i+1, end='')
#List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    lista =[[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i+j+k !=n]
    print(lista, end='')
        
#Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    arr2 =list(set(arr))
    arr2.sort()
    print(arr2[-2])
#Nested lists

if __name__ == '__main__':
    l = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        l.append([name,score])

    sortedlist = sorted(l, key = lambda x: (x[1],x[0]))
    i=1
    n=0
    while sortedlist[n][1] == sortedlist[i][1]:
        i = i + 1
    
     
    s = sortedlist[i][1]
    for r in sortedlist:
        for x in r:
            if x == s:
                print(r[0])
#Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
    print("{:.2f}".format(sum(student_marks[query_name]) / 3))

#Lists
if __name__ == '__main__':
    nums=[]
    N = int(input())
    
    for _ in range(N):
        split_list=input().split()
        if split_list[0] == 'print':
            print(nums)
            
        elif split_list[0] in ['sort','pop','reverse']:
            eval(f"nums.{split_list[0]}()")
            
        elif split_list[0] == 'insert':
            nums.insert(int(split_list[1]),int(split_list[2]))
            
        elif split_list[0] in ['remove','append']:
            eval(f"nums.{split_list[0]}({split_list[1]})")
#Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t=tuple(integer_list)
    print(hash(t))
#Swap Case
def swap_case(s):
    x= ""
    for letter in s: 
        if letter== letter.upper():
            x += letter.lower()
        else: 
            x += letter.upper()
    return x
    
    

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)

    
#String Split and Join

def split_and_join(line):
    
    return line.replace(" ", "-")
    

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)
#What’s your name? 
def print_full_name(first, last):
    
    s= 'Hello ' + first + ' '+ last + '! You just delved into python.' 
    print(s)

#mutations 
def mutate_string(string, position, character):
    return string[:position] + character + string[position+1:]

#find a string 
def count_substring(string, sub_string):
    b= len(string) 
    a= len(sub_string) 
    count=0 
    for i in range(b): 
        if string[i:i+a] == sub_string: 
            count= count + 1 
    return count
#String Validators
if __name__ == '__main__':
    s = input()
    l=list(s)
    print (any(i.isalnum() for i in l))
    print (any(i.isalpha() for i in l))
    print (any(i.isdigit() for i in l))
    print (any(i.islower() for i in l))
    print(any(i.isupper() for i in l))
#Text alignment 

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


#text wrap 
def wrap(string, max_width):
    return "\n".join(textwrap.wrap(string, max_width))


#designer door mat 
n,m = list(map(int, input().split()))
for i in range(1 , n , 2): print( (".|." * i).center( m , "-" ) )
print("WELCOME".center( m , "-" ))
for i in range(n-2 , 0 , -2): print( (".|." * i).center( m , "-" ) )

#string formatting 
def print_formatted(number):
    # your code goes here
    w=len('{:b}'.format(number))
    for i in range(1,number+1): 
        print('{0:{width}n} {0:{width}o} {0:{width}X} {0:{width}b}'.format(i,width=w))

#capitalize 
def solve(s):
    st= s.split(' ')
    new=''
    for i in st: 
        new += i.capitalize() +' '
    return new

#merge the tools 
def merge_the_tools(string, k):
    # your code goes here
    for i in range(0, len(string), k):
        s = ''
        for x in range(i, i+k):
            if string[x] not in s:
                s += string[x]
        print(s)
#introduction to set 
def average(array):
   
    s=set(array)
    media= sum(s) / len(s)
    return media
#no idea!  
a, b = list(map(int, input().split()))
lista= list(map(int, input().split()))
A = set(map(int, input().split()))
B = set(map(int, input().split()))
result = 0
for i in lista:
    if i in A:
        result += 1
    elif i in B:
        result -= 1       

print(result)

#symmetric difference 
M = int(input())
set1 = set(map(int,input().split()))
N = int(input())
set2 = set(map(int,input().split()))
diff1= set1.difference(set2)
diff2=set2.difference(set1)
diff1.update(diff2)
set3= sorted(diff1)
for number in set3: 
    print (number)
#set add
n= int(input())
set1= set()
for i in range(0,n):
    set1.add(input())
print (len(set1))
#Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
command= int(input())
for i in range(0,command):
    cmd=input().split()
    if cmd[0]=='pop': 
        s.pop()
    elif cmd[0]=='discard': 
        s.discard(int(cmd[1]))
    else: 
        s.remove(int(cmd[1]))
print(sum(s))

#Set .union() Operation
n = int(input())
set1= set(map(int, input().split()))
n1= int(input())
set2=set(map(int, input().split()))
print (len(set1|set2))


#Set .intersection() Operation
n=int(input())
A = set(map(int, input().split()))
n1=int(input())
B = set(map(int, input().split()))
A.intersection_update(set(B))
print(len(A))

#Set .difference() Operation
n=int(input())
A = set(map(int, input().split()))
n1=int(input())
B = set(map(int, input().split()))
A.difference_update(B)
print(len(A))

#Set .symmetric_difference() Operation
n=int(input())
a= set(map(int, input().split()))
n1=int(input())
b=set(map(int, input().split()))
a.symmetric_difference_update(b)
print(len(a))

#Set Mutations
n = int(input())
set1 = set(map(int, input().split()))
for i in range(int(input())):
    commandi = list(input().split())
    set2= set(map(int, input().split()))
    if commandi[0] == 'intersection_update':
        set1.intersection_update(set2)
    elif commandi[0]== 'symmetric_difference_update':
        set1.symmetric_difference_update(set2)
    elif commandi[0]=='update':
        set1.update(set2)
    else:
        set1.difference_update(set2)

print(sum(set1))
#The Captain's Room
k= int(input())
lista= list(map(int, input().split()))
set1 = set(lista)
for i in list(set1): 
    lista.remove(i)
set2=set(lista)
set1= set1.difference(set2)
for i in set1:
    print(i)


#Check Subset
cases = int(input())
for i in range(cases): 
    n=int(input())
    set1= set(map(int, input().split()))
    n2= int(input())
    set2= set(map(int, input().split()))
    print(set1.issubset(set2))

#check strict superset 
a, n= set(input().split()), int(input())

print(all(a > set(input().split()) for _ in range(n)))

#collections.Counter()
from collections import Counter

n = int(input())
taglie = list(input().split())
lista= dict(Counter(taglie))
customers = int(input())
count = 0
for _ in range(customers):
    size, price = input().split()
    if size in lista.keys():
        if lista[size] != 0:
            lista[size] -=1
            count+=int(price)  
print(count)

#DefaultDict Tutorial
from collections import defaultdict
numeri=input().split(" ")
n=int(numeri[0])
m=int(numeri[1])
dictt = defaultdict(list)

for x in range(1, n+1):
    chiave = input()
    dictt[chiave].append(x)

for y in range(1, m+1):
    chiave= input()
    if chiave not in dictt: 
            print(-1)
    else: 
        print(" ".join([str(item) for item in dictt[chiave]]))

#Collections.namedtuple()
from collections import namedtuple
N=int(input())
S=namedtuple('Student',input().rsplit())
print(sum([int(S(*input().rsplit()).MARKS) for _ in range(N)])/N)

#Collections.OrderedDict()
from collections import OrderedDict
n= int(input())
order_dict={}
order_dict= OrderedDict()
for _ in range(n):
    lista=input().split()
    item=" ".join(map(str, lista[:-1]))
    prezzo= int(lista[-1])
    
    if item in order_dict:
        order_dict[item] += prezzo
    else: 
        order_dict[item] =prezzo
for item, i in order_dict.items(): 
    print(item, i)
    
#Word Order
from collections import Counter 
n=int(input())
counter = Counter([input() for i in range(n)]) 
print(len(counter)) 
for count in counter: 
    print(counter.get(count), end=' ')

#Collections.deque()
from collections import deque
operations = int(input())
d= deque()
for _ in range(operations):

    lista = [i for i in input().split()]
    
    if lista[0]=='append':
        d.append(int(lista[1]))
    
    elif lista[0]=='appendleft':
        d.appendleft(int(lista[1]))
        
    elif lista[0]=='pop':
        d.pop()
        
    else:
        d.popleft()

for i in d:
    print(i, end=' ')

#Calendar Module
import calendar 
import datetime
month, day, year = map(int, input().split())
data= datetime.date(year, month , day)
print(data.strftime("%A").upper())


#Time Delta
from datetime import datetime 
import math
import os
import random
import re
import sys

def time_delta(t1, t2):
    t3 = datetime.strptime(t1, "%a %d %b %Y %H:%M:%S %z")
    t4 = datetime.strptime(t2, "%a %d %b %Y %H:%M:%S %z")
    return str(abs(int((t3 - t4).total_seconds())))
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()

#exceptions 
cases= int(input())
for i in range(cases):
    try:
        x,y=map(int,input().split())
        print(x//y)
    except Exception as e:
        print("Error Code:",e)

#zipped!
n, m = map(int,input().split())
lista = []
for _ in range(int(m)):
    voti = list(map(float, input().split()))
    lista.append(voti)

for x in zip(*lista):
    print(sum(x)/m)

#ginortS
S = list(input())


first = [i for i in S if i.isalpha() and i.islower()]
second = [i for i in S if i.isalpha() and i.isupper()]
third = [i for i in S if i.isdigit() and not int(i) % 2 == 0]
fourth = [i for i in S if i.isdigit() and int(i) % 2 == 0]
print("".join(sorted(first) + sorted(second) + sorted(third) + sorted(fourth)))

#athlete sort 
import math
import os
import random
import re
import sys
import operator


if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    sorted_list = sorted(arr, key= operator.itemgetter(k))
    for sl in sorted_list:
        print(*sl)
#Map and Lambda Function
cube = lambda x: x**3 # complete the lambda function 

def fibonacci(n):
    n1, n2 = 0,1 
    arr = [n1,n2]
    if n==0:
        return 0
    elif n==1: 
        return n2 
    else: 
        for i in range(2, n):
            c= n1+n2 
            n1=n2
            n2=c
            arr.append(c)
        return (arr)
    # return a list of fibonacci numbers
    
 
    

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))


#Detect Floating Point Number
import re 
n= int(input())
for i in range(n):
    if re.match("^[-+]?[0-9]*\.[0-9]+$", input()):
        print(True)
    else:
        print(False)
#re.split()
regex_pattern = r"[.,]"	# Do not delete 'r'.

import re
print("\n".join(re.split(regex_pattern, input())))

#Group(), Groups() & Groupdict()

import re
x = re.search(r"([a-z A-Z 0-9])\1+", input())
if x: 
    print (x.group()[0])
else: 
    print(-1)


#Re.findall() & Re.finditer()
import re
st = input()
lista = re.findall(r"(?<=[^aeiouAEIOU])([aeiouAEIOU]{2,})(?=[^aeiouAEIOU])", st)
for z in lista:
    print(z)
if len(lista)==0:
    print(-1)


#Re.start() & Re.end()


import re

s=input()
k=input()

if (k not in s):
    print("(-1, -1)")
else: 
    for x in re.finditer(r"(?=("+k+"))", s):
        print(f"({x.start(1)}, {x.end(1)-1})")
#Regex Substitution
import re 
n=int(input())
patt1= r"(?<= )(&&)(?= )"
patt2= r"(?<= )(\|\|)(?= )"
for i in range(n):
    st = input()
    st2 = re.sub(patt1, "and", st)
    print(re.sub(patt2, "or", st2))

#Validating Roman Numerals
regex_pattern = r"(M{0,3})(C[DM]|D?C{0,3})(X[LC]|L?X{0,3})(I[VX]|V?I{0,3})$"	# Do not delete 'r'.

import re
print(str(bool(re.match(regex_pattern, input()))))
#validating phone numbers 
import re 
n= int(input())

for _ in range(n): 
    if re.search(r"^[789][0-9]{9}$", input()):
        print ('YES')
    else:
        print('NO')


#Validating and Parsing Email Addresses
import re

n= int(input())
for _ in range(n):
    name, email = list(map(str, input().split()))
    if bool(re.search( r"<[a-z][a-zA-Z0-9\-\.\_]+\@[a-zA-Z]+\.[a-zA-Z]{1,3}>", email)):
        print (name, email)


#Hex Color Code

import re
n=int(input())
pattern = r'(?<=.)+#[0-9a-fA-F]{3,6}'
for _ in range(n):
    string = input()
    ms = re.findall(pattern, string)
    if len(ms):
        for m in ms:
            print(m)

#HTML Parser - Part 1
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(f"Start : {tag}")
        for name,value in attrs:
            print(f"-> {name} > {value}")

    def handle_startendtag(self, tag, attrs):
        print(f"Empty : {tag}")
        for name,value in attrs:
            print(f"-> {name} > {value}")

    def handle_endtag(self, tag):
        print(f"End   : {tag}")


parser = MyHTMLParser()
n=int(input())
for _ in range(n):
    parser.feed(input())
parser.close()


#HTML Parser - Part 2
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if data != '\n':
            if "\n" in data:
                print(">>> Multi-line Comment")
                print(data)
            else:
                print(">>> Single-line Comment")
                print(data)
    def handle_data(self, data):
        if  data != '\n':
            print(">>> Data")
            print(data)
  
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()
#Detect HTML Tags, Attributes and Attribute Values
from html.parser import HTMLParser
n=int(input())
class CustomHtmlParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for x in attrs:
            print("->", x[0], ">", x[1])


parser = CustomHtmlParser()
for _ in range(n):
    parser.feed(input())
#validating UID
n = int(input())
for i in range(n):
    st = input()
    setst=set(st)
    if len(st) == 10 and len(setst) == 10 and len([i for i in st if i.isnumeric()]) >= 3 and len([i for i in st if i.isupper()]) >= 2:
        print("Valid")
    else:
        print("Invalid")


#xml 1 - find the score 
import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    # your code goes here
    n = len(node.attrib)
    for x in node.findall(r".//"):
        n += len(x.attrib)
    return n
if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))

#xml 2 find the maximum depth 
import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    # your code goes here

if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)

#Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        # complete the function
        f([f"+91 {i[-10:-5]} {i[-5:]}" for i in l])
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 

#Decorators 2 - Name Directory
def person_lister(f):
    def inner(people):
        
        for i in people:
            i[2] = int(i[2])
        people.sort(key=operator.itemgetter(2))
        return [f(person) for person in people]
        
    return inner



#Arrays
import numpy

def arrays(arr):
    
    a=numpy.array(arr, float) 
    return numpy.flip(a)

arr = input().strip().split(' ')
result = arrays(arr)
print(result)
#Shape and Reshape
import numpy
lista = list(map(int, input().split()))   
a=numpy.array(lista)
print(numpy.reshape(a, (3, 3)))


#Transpose and Flatten
import numpy
n, m = map(int, input().split())
lista = []
for _ in range(n):
    lista.append(list(map(int, input().split())))
arr = numpy.array(lista)
print(numpy.transpose(arr))
print(arr.flatten())

#Concatenate
import numpy
n,m,p = list(map(int, input().split()))

l1 = numpy.array([list(map(int, input().split())) for _ in range(n)])
l2 = numpy.array([list(map(int, input().split())) for _ in range(m)])
print(numpy.concatenate((l1, l2)))

#Zeros and Ones
import numpy

n=list(map(int,input().split()))
print(numpy.zeros(n, int))
print (numpy.ones(n, int))



#Eye and Identity
import numpy
numpy.set_printoptions(legacy = '1.13')
n,m = list(map(int, input().split()))
print (numpy.eye(n, m, k = 0))


#Array Mathematics
import numpy
N, M = numpy.array(input().split(), int)
array1 = numpy.array([input().split() for i in range(N)], int)
array2 = numpy.array([input().split() for i in range(N)], int)
ris = [numpy.add(array1, array2), numpy.subtract(array1, array2), numpy.multiply(array1, array2), numpy.floor_divide(array1, array2), numpy.mod(array1, array2), numpy.power(array1, array2)]
for x in ris:
    print(x)



#Floor, Ceil and Rint
import numpy
numpy.set_printoptions(legacy='1.13')

arr= numpy.array(list(map(float,input().split())))

print(numpy.floor(arr), numpy.ceil(arr),numpy.rint(arr),sep="\n")



#sum and prod 
import numpy
n,m = list(map(int, input().split()))
arr = numpy.array([list(map(int, input().split())) for _ in range(n)])
print(numpy.prod(numpy.sum(arr, axis=0)))

#min and max
import numpy
n,m = list(map(int, input().split()))
arr = numpy.array([list(map(int, input().split())) for _ in range(n)])
print(numpy.max(numpy.min(arr, 1)))   

#mean, var and std
import numpy
n,m = list(map(int, input().split()))
arr= numpy.array([list(map(int, input().split())) for _ in range(n)])
print(numpy.mean(arr, 1))
print(numpy.var(arr, 0))
print(round(numpy.std(arr,  None),11))

#dot and cross 
import numpy
a=int(input())
a1=numpy.array([list(map(int,input().split())) for _ in range(a)])
a2=numpy.array([list(map(int,input().split())) for _ in range(a)])

print(numpy.dot(a1,a2))

#inner abd outer 
import numpy

a = numpy.array(list(map(int, input().split())))
b = numpy.array(list(map(int, input().split())))
print (numpy.inner(a, b))
print (numpy.outer(a, b))


#polynomials 
import numpy

P = list(map(float, input().split()))
x = float(input())

print(float(numpy.polyval(P, x)))


#linear algebra 
import numpy

print(round(numpy.linalg.det(numpy.array([list(map(float,input().split())) for _ in range(int(input()))])),2))



#################################


#PROBLEM 2 
#birthday cake candles 
def birthdayCakeCandles(candles):
    # Write your code here
    
    return candles.count(max(candles))
        
  

    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

#kangaroo 
def kangaroo(x1, v1, x2, v2):
    v= v1- v2 
    x= x2 -x1 
    if ((v2 >= v1) or ((x2-x1)%v)) !=0 : 
        return ("NO")
    else:
        return ("YES")

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

#strange advertising 

import math
import os
import random
import re
import sys


def viralAdvertising(n):
    # Write your code here
    like = 2
    somma= 2
    shared=5 
    for i in range(n-1): 
       shared = like*3 
       like= shared//2 
       somma+=like
    return somma


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

#recursive digit sum 
import math
import os
import random
import re
import sys

def superDigit(n, k):
  
    m = int(n) * k % 9
    if m!=0: 
        return m 
    else: 
        return 9
    
   

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

#insertion sort1 


import math
import os
import random
import re
import sys



def insertionSort1(n, arr):
    
    for i in range((n-1),0,-1):
        if arr[i] < arr[i-1]:
            t = arr[i]
            arr[i] = arr[i-1]
            print(*arr)
            arr[i-1] = t
    print(*arr)

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

#insertion sort2 
import math
import os
import random
import re
import sys

def insertionSort2(n, arr):
    
    for i in range(1,n):
        for j in range(i):
            if arr[j] > arr[i]:
                arr[i], arr[j] = arr[j], arr[i]
        print(*arr)

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)





































































