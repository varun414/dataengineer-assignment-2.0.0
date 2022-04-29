# Data Engineer Assignment

This assignment consists of two parts, an architectural design and a coding challenge. 

## Architecture

We have online trading systems that provide real-time data and we want to make 
it available for historical analysis. The data that we want to process and store consists 
of metadata (timestamp of event, source system ID) and payload (stock ID, bid price, ask 
price). This data can be retrieved from the source systems in two different ways:

 - A subscription API (i.e. gRPC)
 - An event log file in CSV format (written every minute on a disk, retained for 1 day)

Note, trading systems are network latency critical and are active while an exchange 
is open i.e 09:00-18:00.

Design a system that will collect events from such trading systems, process and store it for historical 
analysis. We expect a system level design describing the major components and reasons for using them. 
Limit the detail to a single page.


## Spark

Attached is a small PySpark project. There are two datasets, trades and prices, that need to 
be combined in two ways. Examples of the expected output are in the source.

Your answer should be a runnable `main.py` script that shows the results from each join operation,
matching the examples in the docstrings.

Include an evaluation of your solution, how it would scale to 100,000s events over multiple 
days and 100s of ids, and if your approach would be different at that scale.

## Review

Your design, code and evaluation will be reviewed, and a subject for discussion during a 
technical interview.