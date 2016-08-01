import numpy as np

def niceAxisRange(x, margin=0.04):
    mn = min(x)
    mx = max(x)
    d = (mx - mn) * margin
    return (mn-d, mx+d)
