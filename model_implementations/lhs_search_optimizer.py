import numpy as np
from scipy.stats import qmc


class LHS_Search_Optimizer():

    def __init__(self, config_space, n_samples=1000):
        self.search_space = config_space
        self.n_samples = n_samples

        self.generated_configurations = self.generate_lhs_configs_flex()
        self.current_index = 0

    def generate_lhs_configs_flex(self):
        dim = len(self.search_space)
        sampler = qmc.LatinHypercube(d=dim)
        # create n_samples points in the [0, 1)^dim space.
        samples = sampler.random(n=self.n_samples)

        # each sample has [0, len(search space)] points between [0,1]
        configs = []
        for sample in samples:
            config = {}
            for i, param in enumerate(self.search_space):
                p_type = param['type']
                value = None
                if p_type in ['categorical','discrete']:
                    domain = param['domain']
                    n_values = len(domain)
                    # map random sample to paramter value
                    idx = int(np.floor(sample[i] * n_values))
                    # Safety check
                    if idx >= n_values:
                        idx = n_values - 1
                    value = domain[idx]
                elif p_type == 'integer':
                    lower = param['lower']
                    upper = param['upper']
                    # (upper - lower + 1) possible values.
                    range_size = upper - lower + 1
                    idx = int(np.floor(sample[i] * range_size))
                    if idx >= range_size:
                        idx = range_size - 1
                    value = lower + idx
                else:
                    raise ValueError(f"Unknown parameter type {p_type} for parameter {param['name']}.")
                config[param['name']] = value
            configs.append(config)
        return configs

    def reset(self):
        self.generated_configurations = self.generate_lhs_configs_flex()
        self.current_index = 0

    def suggest(self):
        config = self.generated_configurations[self.current_index]
        self.current_index = self.current_index + 1
        return config


    def get_all_configurations(self):
        return self.generated_configurations

    def report(self, config, result):
        pass
