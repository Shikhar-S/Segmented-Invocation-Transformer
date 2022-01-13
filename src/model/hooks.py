import pickle
import os
from main_utils import get_logger
logger = get_logger()

class LoggingHook:
    def __init__(self,module,save_directory):
        super(LoggingHook,self).__init__()
        self.save_path = save_directory
        self.fwd_batch_counters = {}
        self.bkwd_batch_counters = {}
        for name,submodule in module.named_modules():
            logger.info(f'Registering submodule {name}')
            self.fwd_batch_counters[name] = 0
            self.bkwd_batch_counters[name] = 0
            fwd_path = os.path.join(self.save_path,f'{name}_forward')
            bkwd_path = os.path.join(self.save_path,f'{name}_backward')
            os.makedirs(fwd_path,exist_ok=False)
            os.makedirs(bkwd_path,exist_ok=False)
            submodule.register_forward_hook(self.fwd_hook(name))
            submodule.register_full_backward_hook(self.bkwd_hook(name))

    def fwd_hook(self, submodule_name: str):
        def fn(_, inputs, output):
            self.fwd_batch_counters[submodule_name] += 1
            _path = os.path.join(self.save_path,f'{submodule_name}_forward',f'{self.fwd_batch_counters[submodule_name]}.pkl')
            with open(_path,'wb') as f:
                pickle.dump(inputs,f)
                pickle.dump(output,f)
        return fn
    
    def bkwd_hook(self, submodule_name: str):
        def fn(_, grad_input, grad_output):
            self.bkwd_batch_counters[submodule_name] += 1
            _path = os.path.join(self.save_path,f'{submodule_name}_backward',f'{self.bkwd_batch_counters[submodule_name]}.pkl')
            with open(_path,'wb') as f:
                pickle.dump(grad_input,f)
                pickle.dump(grad_output,f)
        return fn