import datetime as dt
import time

start = time.clock()
end = time.clock()
print(end-start)

start=dt.datetime.today()
time.sleep(100)
end=dt.datetime.today()

print(end-start)