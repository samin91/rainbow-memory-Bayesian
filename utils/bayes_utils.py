import torch
torch.use_deterministic_algorithms(True, warn_only=True)

def configure_prior_conversion_function(prior_conv_func_str):
    if prior_conv_func_str == "sqrt":
        return torch.sqrt
    elif prior_conv_func_str == "exp":
        return torch.exp
    elif prior_conv_func_str == "mul2":
        return lambda x : 2*x
    elif prior_conv_func_str == "mul3":
        return lambda x : 3*x
    elif prior_conv_func_str == "mul4":
        return lambda x : 4*x
    elif prior_conv_func_str == "mul8":
        return lambda x : 8*x
    elif prior_conv_func_str == "log":
        return torch.log
    elif prior_conv_func_str == "pow2":
        return lambda x : x**2
    elif prior_conv_func_str == "pow3":
        return lambda x : x**3
    elif prior_conv_func_str == "div":
        return lambda x : 1/x
    elif prior_conv_func_str == "none":
        return lambda x : x
    else:
        raise RuntimeError(f"Function {prior_conv_func_str} is not supported!")
