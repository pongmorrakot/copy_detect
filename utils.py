import subprocess
import numpy as np
from sklearn.preprocessing import normalize as sknormalize


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def min2sec(timestamp):
    time = str(timestamp.decode("utf-8")).strip('\"').split(":")
    time = int(time[0])*60 + int(time[1])
    return time

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    output = result.stdout.split()
    final_out = output[len(output)-1]
    return float(final_out)

def normalize(x, copy = False):
    """
    A helper function that wraps the function of the same name in sklearn.
    This helper handles the case of a single column vector.
    """
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(sknormalize(x.reshape(1, -1), copy = copy))
        #return np.squeeze(x / np.sqrt((x ** 2).sum(-1))[..., np.newaxis])
    else:
        return sknormalize(x, copy = copy)
        #return x / np.sqrt((x ** 2).sum(-1))[..., np.newaxis]

def pad(arr, max_size):
    # print(np.shape(arr))
    # print(max_size)
    x,y = np.shape(arr)
    target = np.zeros((x, max_size))
    # # print(np.shape(target))
    # # print(np.shape(target[:,:y]))
    target[:,:y] = arr
    output = normalize(target)
    # print(np.shape(output))
    return output
