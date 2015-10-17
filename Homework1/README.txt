1. calculates the sleep time for arrival and service on a per consumer/producer loop (each thread).
2. averages the consumer (server) utilization over all servers.
3. calculate the standard deviations, but prints out the sum per item, and the sum is over the number of consumers used.
4. Using nanosleep, and running against the test values (stated at the bottom of the write up) makes the test run a few minutes. I'm not sure this is desired.
5. prints out the current command settings; especially useful if defaults are used.


For compilation, please use the following line: make all
From there, the command line args should work as designed, per the write-up.

Sample output:


jheld@itserver6:~/cs752/cs752_work/Homework1$ ./a.out -N 5 -L 5.1 -M 1.8
num servers: 5, lambda: 5.100000, mu: 1.800000, cust: 1000
Queue length mean: 0.124035
std deviation queue length: 0.514126
sum of mean customer wait time: 0.000006
sum of std deviation customer wait time: -nan
sum of mean service time: 2.730041
sum of std deviation service time: 311.372324
sum of mean arrival time: 0.195185
sum of std deviation arrival time: 112.569809
sum of server utilization: 1.308006
finished joining consumers, total processed: 1000


jheld@itserver6:~/cs752/cs752_work/Homework1$ ./a.out -N 3 -L 5.1 -M 1.8
num servers: 3, lambda: 5.100000, mu: 1.800000, cust: 1000
Queue length mean: 5.397915
std deviation queue length: 5.556690
sum of mean customer wait time: 0.000003
sum of std deviation customer wait time: -nan
sum of mean service time: 1.642624
sum of std deviation service time: 324.229025
sum of mean arrival time: 0.199742
sum of std deviation arrival time: 117.622107
sum of server utilization: 2.703218
finished joining consumers, total processed: 1000


jheld@itserver6:~/cs752/cs752_work/Homework1$ ./a.out 
num servers: 1, lambda: 3.000000, mu: 4.000000, cust: 1000
Queue length mean: 2.260010
std deviation queue length: 2.615856
sum of mean customer wait time: 0.000004
sum of std deviation customer wait time: -nan
sum of mean service time: 0.263467
sum of std deviation service time: 155.128755
sum of mean arrival time: 0.332926
sum of std deviation arrival time: 192.623698
sum of server utilization: 0.750262
finished joining consumers, total processed: 1000
