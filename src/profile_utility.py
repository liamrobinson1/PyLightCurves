import time

global tstart


def tic():
    global tstart
    tstart = time.perf_counter_ns()


def toc():
    tend = time.perf_counter_ns()
    telapsed = (tend - tstart) / 1e9
    print(f"Elapsed time: {telapsed:.2e} seconds")
    return telapsed
