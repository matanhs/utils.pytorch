from inspect import signature
from copy import deepcopy
from collections import OrderedDict
from abc import abstractmethod, ABCMeta
import warnings


class BaseConfigurationGroup():
    def __init__(self,name,matcher,builder=None):
        if not isinstance(matcher,BaseMatcher):
            warnings.warn('matcher is not a subclass of BaseMatcher')
        self.name=name
        self.matcher=matcher
        self.builder=builder

class BaseMatcher(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, module, trace_name, **kwargs):
        pass

class BasicTypeMatcher(BaseMatcher):
    def __init__(self, target_class):
        self.target_class = target_class

    def __call__(self, module, trace_name, **kwargs):
        return isinstance(module, self.target_class)

class FirstNMatcher(BaseMatcher):
    def __init__(self,limit,matcher):
        self.matcher = matcher
        self.limit = limit
        self.invoke_count = 0

    def __call__(self, module, trace_name):
        if self.invoke_count < self.limit:
            match = self.matcher(module, trace_name)
            if match:
                self.invoke_count+=1
                return True
        return False

class ExactAttrMatcher(BaseMatcher):
    def __init__(self,target_class,kw_attributes):
        assert isinstance(kw_attributes, dict),'please provide a dictionary of attributes and values'
        self.attrs_to_match = kw_attributes
        self.target_class = target_class

    def __call__(self, module, trace_name, **kwargs):
        if isinstance(module, self.target_class):
            for attr,value in self.attrs_to_match.items():
                if getattr(module,attr) != value:
                    return False
            return True

## this is a basic rewriter class that can be extended for general purpose module replacement
class ReWriter():
    _SUPPORTED_MODULES = []

    def __init__(self,**config):
        self._verbose=config.get('verbose',0)
        #self._fallback_matchers = [BasicTypeMatcher(C) for C in type(self)._SUPPORTED_MODULES]
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
                #print(long_name)
                match_found = False
                for group_name,(matcher_fn,builder_fn) in self.group_fns.items():
                    if matcher_fn(c,long_name):
                        if self._verbose>0:
                            print(f'matched {c}: {signature(c.__class__.__init__).parameters.keys()} with {group_name} group')
                        match_found=True
                        replace_children_dict[cn]=builder_fn(c)
                        break

                ## note: assumes matched module's children are replaced by the builder_fn so we dont need to
                # handle them in the recursion
                if not match_found:
                    _recurse_module(c, long_name)

            if len(replace_children_dict)>0:
                if self._verbose > 1:
                    print(f'replacing {long_name} modules',replace_children_dict)
                for cn, new_module in replace_children_dict.items():
                    m._modules[cn]=new_module

        _recurse_module(module)

    ## this method can be used as fallback to handle arbitrary module substitution
    # however it is best to pass responsibility to the builder_fn to correctly build the
    # new module
    @staticmethod
    def _gen_substitute_module(m,init_module_fn):
        ## assume builder_fn constructs a valid subclass of the given module
        # todo: consider passing responsibility for attrs/param/buffers to the builder_fn
        pks = signature(m.__class__.__init__).parameters.keys()
        kwargs={}
        for param_key in pks:
            if param_key=='self':
                continue
            kwargs[param_key]=getattr(m,param_key) if param_key!='bias' else getattr(m,param_key) is not None
        new_module=init_module_fn(**kwargs)
        new_module._parameters=deepcopy(m._parameters)
        new_module._buffers=deepcopy(m._parameters)
        return new_module

    def gen_builder_fn(self,init_module_fn):
        return lambda target_module_instance: type(self)._gen_substitute_module(target_module_instance,init_module_fn)