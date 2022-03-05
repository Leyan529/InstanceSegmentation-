import os
import re
import random
import torch
import json

__all__ = ["save_ckpt", "Meter"]

dirname = os.path.dirname(__file__)
json_file = os.path.join(dirname, "gpu_info.json")

def sort(d, tmp={}):
    for k in sorted(d.keys()):
        if isinstance(d[k], dict):
            tmp[k] = {}
            sort(d[k], tmp[k])
        else:
            tmp[k] = d[k]
    return tmp
    
def collect_gpu_info(model_name, fps):
    fps = [round(i, 2) for i in fps]
    if os.path.exists(json_file):
        gpu_info = json.load(open(json_file))
    else:
        gpu_info = {}
    
    prop = get_gpu_prop()
    name = prop[0]["name"]
    check = [p["name"] == name for p in prop]
    if all(check):
        count = str(len(prop))
        if name in gpu_info:
            gpu_info[name]["properties"] = prop[0]
            perf = gpu_info[name]["performance"]
            if count in perf:
                if model_name in perf[count]:
                    perf[count][model_name].append(fps)
                else:
                    perf[count][model_name] = [fps]
            else:
                perf[count] = {model_name: [fps]}
        else:
            gpu_info[name] = {"properties": prop[0], "performance": {count: {model_name: [fps]}}}

        gpu_info = sort(gpu_info)
        json.dump(gpu_info, open(json_file, "w"))
    return gpu_info

def get_gpu_prop(show=False):
    ngpus = torch.cuda.device_count()
    
    properties = []
    for dev in range(ngpus):
        prop = torch.cuda.get_device_properties(dev)
        properties.append({
            "name": prop.name,
            "capability": [prop.major, prop.minor],
            "total_momory": round(prop.total_memory / 1073741824, 2), # unit GB
            "sm_count": prop.multi_processor_count
        })
       
    if show:
        print("cuda: {}".format(torch.cuda.is_available()))
        print("available GPU(s): {}".format(ngpus))
        for i, p in enumerate(properties):
            print("{}: {}".format(i, p))
    return properties

def save_ckpt(model, optimizer, epochs, ckpt_path, **kwargs):
    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["optimizer"]  = optimizer.state_dict()
    checkpoint["epochs"] = epochs
        
    for k, v in kwargs.items():
        checkpoint[k] = v
        
    prefix, ext = os.path.splitext(ckpt_path)
    # ckpt_path = "{}-{}{}".format(prefix, epochs, ext)
    ckpt_path = "{}{}".format(prefix, ext)
    torch.save(checkpoint, ckpt_path)
    
    
class TextArea:
    def __init__(self):
        self.buffer = []
    
    def write(self, s):
        self.buffer.append(s)
        
    def __str__(self):
        return "".join(self.buffer)

    def get_AP(self):
        result = {"bbox AP": 0.0, "mask AP": 0.0}
        
        txt = str(self)
        values = re.findall(r"(\d{3})\n", txt)
        if len(values) > 0:
            values = [int(v) / 10 for v in values]
            result = {"bbox AP": values[0], "mask AP": values[12]}
            
        return result
    
    
class Meter:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:sum={sum:.2f}, avg={avg:.4f}, count={count}"
        return fmtstr.format(**self.__dict__)
    
                
