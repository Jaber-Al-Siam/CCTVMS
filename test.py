from collections import deque

dq = deque(maxlen=5)

for i in range(10):
    dq.append(i)

for x in dq:
    print(x)
