


from .core import Policy

import torch
import numpy as np
import math



def get_shape_info(named_param_dict):
    shape_info = dict()
    current_index = 0 
    for name,param in named_param_dict.items():
        start_end_indicies = (current_index,current_index + param.data.numel())
        current_index += param.data.numel()
        param_shape = param.data.shape
        shape_info[name] = {"indices" : start_end_indicies,"shape" : param_shape }
    return shape_info

def get_params_flat(named_param_dict):
    shape_info = dict()
    current_index = 0 
    param_views = []
    for name,param in named_param_dict.items():
        start_end_indicies = (current_index,current_index + param.data.numel())
        current_index += param.data.numel()
        param_shape = param.data.shape
        shape_info[name] = {"indices" : start_end_indicies,"shape" : param_shape }
        param_views.append(param.data.view(-1))

    return torch.cat(param_views),shape_info  # NOTE the cat will create a copy...

def overwrite_params_with_flat(model,flat_vec,shape_info):
    model_dict = dict(model.named_parameters())
    for key,info in shape_info.items():
        start_i,end_i = info["indices"]     
        model_dict[key].data = flat_vec[start_i:end_i].view(*info["shape"])



# This is a reimplementation of the cheetah policy in torch
# In theory it should be swappable with the tf one
# Now takes 2 extra optional parameters:  extra_args and device

class CheetahPolicyTorch(Policy,torch.nn.Module):

    needs_stats = True  

    def __init__(
        self,
        ob_space_shape,
        num_actions,
        action_low,
        action_high,
        ac_noise_std=0.01,
        hidden_dims=(256, 256),
        gpu_mem_frac=0.2,
        extra_args=None,
        single_threaded=False,
        theta=None,
        seed=42,
        device=None):

        self.device = device
        if self.device is None:
            self.device = torch.device("cpu")

        super(CheetahPolicyTorch, self).__init__()

        #if single_threaded is True: # so when doing distributed computing it does not spawn multiple threads
        torch.set_num_threads(1)

        self.ob_space_shape = ob_space_shape
        self.num_actions = num_actions
        self.action_low = action_low
        self.action_high = action_high
        self.ac_noise_std = ac_noise_std
        self.hidden_dims = hidden_dims
        if "hidden_dims" in extra_args:
            self.hidden_dims = extra_args["hidden_dims"]
            hidden_dims = None # make sure not using the old one accidentally

        self.gpu_mem_frac = gpu_mem_frac
        self.extra_args = extra_args

        self.random_state = np.random.RandomState(seed)
        self.nonlin = torch.tanh

        assert len(ob_space_shape) == 1
        assert np.all(np.isfinite(action_low)) and np.all(
            np.isfinite(action_high)
        ), "Action bounds required"

        print('Creating torch policy...')

        self.ob_mean = None
        self.ob_std = None
        self.normalization_required = True  # default is true
        if extra_args is not None and "normalization_required" in extra_args:
            self.normalization_required = extra_args["normalization_required"]

        # Create layers
        self.layers = torch.nn.ModuleList()
        previous_dim = self.ob_space_shape[0]
        for hidden_dim in self.hidden_dims:
            self.layers.append(torch.nn.Linear(previous_dim, hidden_dim))
            previous_dim = hidden_dim
        # add final layer
        self.layers.append(torch.nn.Linear(previous_dim, self.num_actions))

        self.shape_info = get_shape_info(dict(self.named_parameters()))

        self.to(self.device)

        if theta is not None:
            self.set_theta(theta)

        

        
    def seed(self, seed):
        self.random_state.seed(seed)

    def forward_internal(self,observation,need_grad=False,custom_parameters=None):  # needs_grad does not do anything, just so the signiture is the same with the other class
        # i believe we are supplied with a list of np arrays
        # we want to transform them to a torch array of shape (batch_size,Observation_size)
        
        if torch.is_tensor(observation) is False:
            observation = torch.stack([torch.from_numpy(ob).float() for ob in observation])
        observation = observation.to(self.device)

        if self.normalization_required is True and self.ob_mean is None:
            raise "ob_mean not set!, did you call set_stats()??"

        # Normalize and clamp
        if self.normalization_required is True:
            x = (observation - self.ob_mean) / self.ob_std
            x = torch.clamp(x, -5.0, 5.0)
        else:
            x = observation

        for layer_i, layer in enumerate(self.layers):
            if custom_parameters is None:
                x = layer(x)
            else:
                x = torch.nn.functional.linear(x, custom_parameters["layers." + str(layer_i) + ".weight"], custom_parameters["layers." + str(layer_i) + ".bias"])
            if layer_i != (len(self.layers)-1):  # dont apply nonlin in the last layer
                x = self.nonlin(x)

        if self.ac_noise_std != 0:
            x += torch.from_numpy(self.random_state.randn(*x.shape)).float().to(self.device) * self.ac_noise_std

        return x

    def forward(self,observation):

        with torch.no_grad():
            x = self.forward_internal(observation)
            return x.numpy()

    def act(self, states):
        return self.forward(states)

    def set_stats(self, ob_mean, ob_std):
        self.ob_mean = torch.from_numpy(ob_mean).float().to(self.device) 
        self.ob_std = torch.from_numpy(ob_std).float().to(self.device) 


    def set_theta(self, theta):
        # TODO might need to torchify
        theta = torch.from_numpy(theta).float().to(self.device) 
        overwrite_params_with_flat(self,theta,self.shape_info)

    def get_theta(self):
        # TODO might need to numpyify
        params_flat,shape_info = get_params_flat(dict(self.named_parameters()))
        return params_flat
    

    def serialize(self):
        # NOTE: not serializing theta; needs to be passed separately
        return super()._serialize(
            self.ob_space_shape,
            self.num_actions,
            self.action_low,
            self.action_high,
            self.ac_noise_std,
            self.hidden_dims,
            self.gpu_mem_frac,
            self.extra_args,
        )

    def after_weight_update(self):
        pass  # nothing to do here, this is needed only for the weight generator net



