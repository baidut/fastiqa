import json # save settings

######### save configuation information #########################################

def serialize(obj):
    """JSON serializer for objects not serializable by default json code"""
    # if isinstance(obj, (datetime, date)):
    #     return obj.isoformat()
    # https://stackoverflow.com/questions/19628421/how-to-check-if-str-is-implemented-by-an-object
    # print(f'{type(obj)}')
    if 'core' in f'{type(obj)}' or 'fastai' in f'{type(obj)}':
        if hasattr(obj, 'name'):
            return obj.name
        else:
            return f'{obj}'
    if obj.__class__.__name__ == 'PosixPath': # type(obj).__str__ is not object.__str__: # pathlib
        return str(obj)
    if obj.__class__.__name__ == 'function':
        return obj.__name__
    if hasattr(obj, 'to_json'):
        return obj.to_json()
    if hasattr(obj, '__dict__'):
        # d = {k:(obj.__dict__[k]) for k in obj.__dict__ if not k.startswith('_')}
        #d = {k:json.dumps((obj.__dict__[k]), default=serialize) for k in obj.__dict__ if not k.startswith('_')}
        d = {}
        for k in obj.__dict__:
            if not k.startswith('_'):
                # print(f'{obj.__class__.__name__}.{k}')
                d[k] = obj.__dict__[k] #json.dumps((obj.__dict__[k]), default=serialize)
        d['__class__.__name__'] = obj.__class__.__name__
        return d
    else:
        return '--not serializable--'
    # return {k:(obj.k) for k in dir(obj) if not k.startswith('_')}
    # raise TypeError ("Type %s not serializable" % type(obj))

def to_json(self, file=None):
    if file is None:
        return json.dumps(self, default=serialize)
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(self, f, default=serialize, ensure_ascii=False, indent=4)
