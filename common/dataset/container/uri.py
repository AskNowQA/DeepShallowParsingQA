class URI:
    def __init__(self, raw_uri):
        self.raw_uri = raw_uri.strip('<>')
        self.label = self.raw_uri[self.raw_uri.rindex('/') + 1:].lower()
