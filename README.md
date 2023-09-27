## Scripts for running a tuning server

Creates a queue at at a provided IP address and port. This queue can be used to push configurations to it and then tuning agents that are running on different servers can pull configs from this queue and run then on the respective machine.

### Start the queue

Start a tmux session one of the server and run the file queue_server.py to initialize a queue.

```python
python queue_server.py
```

### Put cnfigurations in the queue

```python
python tune.py --ip <ip_address_of_queue> --port <port_of_queue> -t enque --name <project_name> --clear
```

`--clear` clears the queues and inserts the command from tune.py

### Create a running client

```
python tune.py --ip <ip_address_of_queue> --port <port_of_queue> -t run --gpu 0
```
