from utils.partial_class import partial_class
from inspect import signature
from copy import deepcopy
from collections import OrderedDict

## this is a basic rewriter class that can be extended for general purpose module replacement
class ReWriter():
    _SUPPORTED_MODULES = []

    _BASIC_MATCHER_FN=lambda C: lambda m,n: isinstance(m, C)

    def __init__(self,**config):
        self._verbose=config.get('verbose',1)
        self._fallback_matchers = [type(self)._BASIC_MATCHER_FN(C) for C in type(self)._SUPPORTED_MODULES]
        self._default_cfgs = OrderedDict()

        cfg_groups=config.get('cfg_groups',OrderedDict())
        cfg_groups.update(self._default_cfgs)

        ## eg. first and last fc layers quant configs, matcher gets the modules' long name and the module itself
        # and returns True if belongs to the group :
        self.group_fns=OrderedDict()
        for group_name,group_cfgs in cfg_groups.items():
            self.group_fns[group_name]=group_cfgs['matcher_fn'],group_cfgs['module_generator_fn']

    ## after rewriter initialization, rewriter call will replace modules by calling every match_fn until the first match
    # is found (order/priority consideration is needed in case multiple group match are possible)
    def __call__(self,module):
        def _recurse_module(m,parent_name=None):
            replace_children_dict={}
            long_name=parent_name or ''
            for cn, c in m.named_children():
                if parent_name == None:
                    long_name = cn
                else:
                    long_name = f'{parent_name}.{cn}'
                print(long_name)
                match_found = 0
                for matcher_fn,builder_fn in self.group_fns.values():
                    if matcher_fn(c,long_name):
                        if self._verbose>1:
                            print('matched',c,builder_fn.source_class,signature(builder_fn).parameters.keys())
                        match_found=1
                        replace_children_dict[cn]=type(self)._gen_substitute_module(c,builder_fn)
                        ## todo got the replacement module, need to set it instead of the original
                        break

                ## note: assumes matched modules children are replaced by the builder_fn so we dont need to
                # handle them in the recursion
                if not match_found:
                    _recurse_module(c, long_name)

            if len(replace_children_dict)>0:
                if self._verbose > 0:
                    print(f'replacing {long_name} modules',replace_children_dict)
                for cn, new_module in replace_children_dict.items():
                    m._modules[cn]=new_module

        _recurse_module(module)

    ## this method can be used as fallback to handle arbitrary module substitution
    # however it is best to pass responsibility to the builder_fn to correctly build the
    # new module
    @staticmethod
    def _gen_substitute_module(m,builder_fn):
        ## assume builder_fn constructs a valid subclass of the given module
        # todo: consider passing responsibility for attrs/param/buffers to the builder_fn
        pks = signature(m.__class__.__init__).parameters.keys()
        kwargs={}
        for param_key in pks:
            if param_key=='self':
                continue
            kwargs[param_key]=getattr(m,param_key) if param_key!='bias' else getattr(m,param_key) is not None
        new_module=builder_fn(**kwargs)
        new_module._parameters=deepcopy(m._parameters)
        new_module._buffers=deepcopy(m._parameters)
        return new_module
