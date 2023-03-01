import time

global tstart


def tic():
    """Starts a timer, ended by toc()"""
    global tstart
    tstart = time.perf_counter_ns()


def toc():
    """Ends the timer, returns the elapsed time"""
    tend = time.perf_counter_ns()
    telapsed = (tend - tstart) / 1e9
    print(f"Elapsed time: {telapsed:.2e} seconds")
    return telapsed
