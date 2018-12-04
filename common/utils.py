try:
    import builtins

    profile = builtins.__dict__['profile']
except KeyError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func


class Utils:


    @staticmethod
    @profile
    def ngrams(string, n=3):
        ngrams = zip(*[string[i:] for i in range(n)])
        return set([''.join(ngram) for ngram in ngrams])
