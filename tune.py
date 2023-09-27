import os, sys, argparse, time
import requests
import json
from requests import get
import socket
import copy


class QueueClient:
    def __init__(self, url):
        self.queueURL = url

    def dequeServer(self):
        try:
            req = requests.get(self.queueURL + "query")

            if req.status_code == 200:
                data = json.loads(req.text)
                if "Error" in data:
                    return -1
                else:
                    return data["data"]
            else:
                print("Error! Status code :" + str(req.status_code))
                return -1

        except Exception as e:
            print("Error in Queue Server!! \n\n", e)
            return -1

    def enqueue_list(self, data_list):
        try:
            data = {"data": data_list}
            headers = {"Content-Type": "application/json"}
            req = requests.post(
                self.queueURL + "enqueue_list", data=json.dumps(data), headers=headers
            )

            msg = json.loads(req.text)
            return msg

        except Exception as e:
            print("Error in Queue Server!! \n\n", e)

    def enqueue(self, data):
        try:
            data = {"data": data}
            headers = {"Content-Type": "application/json"}
            req = requests.post(
                self.queueURL + "enqueue", data=json.dumps(data), headers=headers
            )

            msg = json.loads(req.text)
            return msg

        except Exception as e:
            print("Error in Queue Server!! \n\n", e)

    def initServer(self, data_list):
        try:
            data = {"list": data_list}
            headers = {"Content-Type": "application/json"}
            req = requests.post(
                self.queueURL + "init", data=json.dumps(data), headers=headers
            )
            msg = json.loads(req.text)
            return msg

        except Exception as e:
            print("Error in Queue Server!! \n\n", e)

    def checkQueue(self):
        try:
            req = requests.get(self.queueURL + "check")
            return json.loads(req.text)
        except Exception as e:
            print("Error in Queue Server!! \n\n", e)

    def getSize(self):
        try:
            req = requests.get(self.queueURL + "size")
            return json.loads(req.text)["Size"]
        except Exception as e:
            print("Error in Queue Server!! \n\n", e)

    def isEmpty(self):
        try:
            req = requests.get(self.queueURL + "check")
            resp = json.loads(req.text)
            return len(resp["Queue"]) == 0
        except Exception as e:
            print("Error in Queue Server!! \n\n", e)

    def clear(self):
        try:
            req = requests.get(self.queueURL + "clear")
            resp = json.loads(req.text)
            print(resp)
            return True
        except Exception as e:
            print("Error in Queue Server!! \n\n", e)
            return False


def enque(args, q):
    if args.get_size:
        print(f"\nTotal {q.getSize()} in queue.")
        exit(0)
    if args.clear:
        q.clear()
    if args.allclear:
        q.clear()
        exit(0)

    exclude_ids = set([])

    src_file = "./src/bert_eval.py"
    project_name = f"{args.name}"
    _task = ["cola", "mrpc", "qqp", "rte", "wnli", "sst2", "mnli", "qnli"]
    _model_name_or_path = ["bert-base-uncased"]
    _peft = ["lora", "ia3", ""]
    _k = [5, 10, 20, 30, 50, 100]
    _replace_factor = [
        -1,
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        8.0,
        10.0,
        12.0,
        14.0,
        15.0,
        20.0,
    ]

    i, count = 0, 0

    for model_name_or_path in _model_name_or_path:
        for task in _task:
            for peft in _peft:
                for k in _k:
                    for replace_factor in _replace_factor:
                        config = {
                            "src_file": src_file,
                            "model_name_or_path": model_name_or_path,
                            "task": task,
                            "peft": peft,
                            "k": k,
                            "project_name": project_name,
                            "replace_factor": replace_factor,
                        }

                        if i not in exclude_ids:
                            count += 1
                            q.enqueue(config)
                            print(config)
                            print(
                                "Inserting {}".format(count),
                                end="\r",
                            )

                            i += 1

    print("\nInserted {}, Total {} in queue. Complete".format(count, q.getSize()))


def get_cmd(q):
    config = q.dequeServer()
    org_config = copy.deepcopy(config)

    if config == -1:
        print("All Jobs Over!!!!")
        time.sleep(10)
        return get_cmd(q)

    cmd = f"{config['src_file']}"
    del config["src_file"]

    if "peft" in config.keys():
        if config["peft"] != "":
            cmd = f"{cmd} --peft={config['peft']}"
        del config["peft"]

    if "train" in config.keys():
        if config["train"]:
            cmd = f"{cmd} --train"
        del config["train"]

    if "test" in config.keys():
        if config["test"]:
            cmd = f"{cmd} --test"
        del config["test"]

    if len(config) > 0:
        for key, value in config.items():
            cmd = f"{cmd} --{key}={value}"

    return cmd, org_config


def gpu_run(q, gpu, only_print=False):
    fail_counter = 0
    while True:
        cmd, org_config = get_cmd(q)
        servername = socket.gethostname().split(".")[0]
        assert "nlp" in servername, f"Invalid server name: {servername}"
        if servername in [
            "nlp15",
            "nlp16",
            "nlp17",
            "nlp18",
            "nlp19",
            "nlp20",
            "nlp21",
        ]:
            cmd = f"export CUDA_VISIBLE_DEVICES={gpu}; cd /nas-ssd2/prateek/projects/peft_pruning; /playpen/prateek/anaconda3/envs/ft_lora/bin/python {cmd}"
        else:
            cmd = f"export CUDA_VISIBLE_DEVICES={gpu}; cd /nas-ssd2/prateek/projects/peft_pruning; /playpen2/home/prateek/anaconda3/envs/ft_lora/bin/python {cmd}"
        print("Command: {}".format(cmd))
        if not only_print:
            return_code = os.system(cmd)
            if return_code != 0:
                fail_counter += 1
                q.enque(org_config)
                if fail_counter >= 15:
                    raise ValueError("Too many failures")
                    exit(1)


def run_args(parser):
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--only_print", action="store_true")
    return parser


def enque_args(parser):
    parser.add_argument("--name", required=True)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--allclear", action="store_true")
    parser.add_argument("--get_size", action="store_true")
    return parser


if __name__ == "__main__":
    ## Start a tmux session in nlp18 and run the file /home/praty/queue/queue_server.py to start the queue
    # Put cnfigurations in the queue.
    ## python tune.py -t enque --name ft_bert --clear
    # print the commands in the queue
    ## python tune.py -t run --gpu 0 --only_print

    try:
        from subprocess import DEVNULL
    except ImportError:
        import os

        DEVNULL = open(os.devnull, "wb")

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", "--type", required=True)
    parser.add_argument("-i", "--ip", required=True, type=str)
    parser.add_argument("-p", "--port", type=int, default=7979)
    args, _ = parser.parse_known_args()

    current_server_ip = get("https://api.ipify.org").content.decode("utf8")
    print(f"IP: http://{args.ip}:{args.port}/")
    q = QueueClient(f"http://{args.ip}:{args.port}/")

    if args.type == "enque":
        # python tune.py -t enque --name ft_bert --clear
        parser = enque_args(parser)
        args, remaining = parser.parse_known_args()
        enque(args, q)
    elif args.type == "run":
        # python tune.py -t run --gpu 0 --only_print
        parser = run_args(parser)
        args, remaining = parser.parse_known_args()
        gpu_run(q, args.gpu, args.only_print)
    else:
        raise ValueError(f"Invalid type: {args.type}")
