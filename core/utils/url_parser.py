'''
'''


def parse_url(full_address):
    kwargs = {}
    try:
        split_colons = full_address.split(':')
        kwargs['port'] = int(split_colons[-1].split('/')[0])
        kwargs['hostname'] = split_colons[-2].split('/')[-1]
        kwargs['address'] = full_address
    except Exception as e:
        print(e)
    return ParseResult(**kwargs)


class ParseResult:

    def __init__(self, **kwargs):
        self.address = kwargs.get('address', '')
        self.hostname = kwargs.get('hostname', -1)
        self.port = kwargs.get('port', '')
