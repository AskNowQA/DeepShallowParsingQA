try:
    import builtins
    profile = builtins.__dict__['profile']
except KeyError:
    # No line profiler, provide a pass-through version
    def profile(func): return func