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

    @staticmethod
    def rgb(red, green, blue):
        """
        Calculate the palette index of a color in the 6x6x6 color cube.
        The red, green and blue arguments may range from 0 to 5.
        """
        red = int(red * 5)
        green = int(green * 5)
        blue = int(blue * 5)
        return 16 + (red * 36) + (green * 6) + blue

    @staticmethod
    def gray(value):
        """
        Calculate the palette index of a color in the grayscale ramp.
        The value argument may range from 0 to 23.
        """
        return 232 + value

    @staticmethod
    def set_color(fg=None, bg=None):
        """
        Print escape codes to set the terminal color.
        fg and bg are indices into the color palette for the foreground and
        background colors.
        """
        if fg:
            print('\x1b[38;5;%dm' % fg, end='')
        if bg:
            print('\x1b[48;5;%dm' % bg, end='')

    @staticmethod
    def reset_color():
        """
        Reset terminal color to default.
        """
        print('\x1b[0m', end='')

    @staticmethod
    def print_color(*args, fg=None, bg=None, **kwargs):
        """
        Print function, with extra arguments fg and bg to set colors.
        """
        Utils.set_color(fg, bg)
        print(*args, **kwargs)
        Utils.reset_color()
