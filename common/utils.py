import os
import urllib
import requests
import ujson as json
import logging.config

try:
    import builtins

    profile = builtins.__dict__['profile']
except KeyError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func


class KB(object):
    def __init__(self, endpoint, default_graph_uri=""):
        self.endpoint = endpoint
        self.default_graph_uri = default_graph_uri
        self.type_uri = "type_uri"
        self.server_available = self.check_server()

    def check_server(self):
        payload = {'query': 'select distinct ?Concept where {[] a ?Concept} LIMIT 1', 'format': 'application/json'}
        try:
            query_string = urllib.parse.urlencode(payload)
            url = self.endpoint + '?' + query_string
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except:
            return False
        return False

    def query(self, q):
        payload = {'query': q, 'format': 'application/json'}
        try:
            query_string = urllib.parse.urlencode(payload)
            url = self.endpoint + '?' + query_string
            r = requests.get(url)
        except:
            return 0, None

        return r.status_code, r.json() if r.status_code == 200 else None


class Cache:
    def __init__(self, file_path):
        self.dic = {}
        self.file_path = file_path
        if os.path.isfile(file_path):
            with open(self.file_path, 'r') as file_handler:
                self.dic = json.load(file_handler)
        self.dirty_counter = 0

    def has(self, key):
        return key in self.dic

    def get(self, key):
        return self.dic[key]

    def add(self, key, value):
        self.dic[key] = value
        self.dirty_counter += 1
        if self.dirty_counter == 10:
            self.save()
            self.dirty_counter = 0

    def save(self):
        with open(self.file_path, 'w') as file_handler:
            json.dump(self.dic, file_handler)


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

    @staticmethod
    def setup_logging(
            default_path='logging.json',
            default_level=logging.INFO,
            env_key='LOG_CFG'
    ):
        """Setup logging configuration
        """
        path = default_path
        value = os.getenv(env_key, None)
        if value:
            path = value
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = json.load(f)
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)

    @staticmethod
    def relations_connecting_entities(ent1, ent2, cache_path):
        if not hasattr(Utils, 'relations_connecting_entities_cache'):
            Utils.relations_connecting_entities_cache = Cache(cache_path)
        key = ent1 + ':' + ent2
        if not Utils.relations_connecting_entities_cache.has(key):
            kb = KB('http://sda01dbpedia:softrock@131.220.9.219/sparql')
            head = 'SELECT DISTINCT ?p1 ?p2 where {{ '
            tail = 'FILTER ((regex(str(?p1), "dbpedia", "i")) && (!regex(str(?p1), "wiki", "i")) && (!regex(str(?p2), "wiki", "i")) && (!regex(str(?p2), "isCitedBy", "i")))}} limit 1000'
            templates = ['<{ent1}>  ?p1  ?s1 .  <{ent2}>  ?p2  ?s1 . ',
                         '?s1  ?p1  <{ent1}> .  ?s1  ?p2  <{ent2}> . ']
            p1s, p2s = [], []
            for item in templates:
                sparql = head + item.format(ent1=ent1, ent2=ent2) + tail
                output = kb.query(sparql)
                if output[0] == 200:
                    if len(output[1]['results']['bindings']) > 0:
                        rel1 = set([item['p1']['value'] for item in output[1]['results']['bindings']])
                        rel2 = set([item['p2']['value'] for item in output[1]['results']['bindings']])
                        p1s.extend(rel1)
                        p2s.extend(rel2)
            # candidate_relations = [set([t for item in candidate_relations for t in item[0] if len(t) > 0]),
            #                        set([t for item in candidate_relations for t in item[1] if len(t) > 0])]
            candidate_relations = [p1s, p2s]
            Utils.relations_connecting_entities_cache.add(key, candidate_relations)
        return Utils.relations_connecting_entities_cache.get(key)

    @staticmethod
    def relation_connecting_entities(ent1, ent2, cache_path):
        if not hasattr(Utils, 'relation_connecting_entities_cache'):
            Utils.relation_connecting_entities_cache = Cache(cache_path)
        key = ent1 + ':' + ent2
        if not Utils.relation_connecting_entities_cache.has(key):
            kb = KB('http://sda01dbpedia:softrock@131.220.9.219/sparql')
            head = 'SELECT DISTINCT ?p1 where { '
            tail = 'FILTER ((regex(str(?p1), "dbpedia", "i")) && (!regex(str(?p1), "wiki", "i")) )} limit 1000'
            templates = ['<{ent1}>  ?p1  <{ent2}>. ',
                         '<{ent2}>  ?p1  <{ent1}>. ']
            candidate_relations = []
            for item in templates:
                sparql = head + item.format(ent1=ent1, ent2=ent2) + tail
                output = kb.query(sparql)
                if output[0] == 200:
                    if len(output[1]['results']['bindings']) > 0:
                        rels = set([item['p1']['value'] for item in output[1]['results']['bindings']])
                        candidate_relations.extend(rels)
            Utils.relation_connecting_entities_cache.add(key, candidate_relations)
        return Utils.relation_connecting_entities_cache.get(key)

    @staticmethod
    def call_web_api(endpoint, raw_input=None, use_json=True, use_url_encode=False, parse_response_json=True,
                     timeout=60):
        proxy_handler = urllib.request.ProxyHandler({})
        if 'sda-srv' in endpoint or '127.0.0.1' in endpoint:
            opener = urllib.request.build_opener(proxy_handler)  # urllib2.build_opener(proxy_handler)
        else:
            opener = urllib.request.build_opener()
        req = urllib.request.Request(endpoint)
        if use_json:
            input = json.dumps(raw_input)
            input = input.encode('utf-8')
            req.add_header('Content-Type', 'application/json')
        elif use_url_encode:
            input = urllib.parse.urlencode(raw_input)
        else:
            input = raw_input
        try:
            response = opener.open(req, data=input, timeout=timeout)
            response = response.read()
            if parse_response_json:
                return json.loads(response)
            else:
                return response
        except Exception as expt:
            print(endpoint)
            print(expt)
            return None
