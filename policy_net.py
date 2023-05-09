"""
Policy network
"""

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import copy


class PolicyNetwork(TorchModelV2, torch.nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name, *args, **kwargs):

        # Call super class constructors
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_config, name)
        torch.nn.Module.__init__(self)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Process keyword arguments
        self.core_hidden_dim = kwargs.get("core_hidden_dim")
        self.head_hidden_dim = kwargs.get("head_hidden_dim")

        input_dim = observation_space.shape[0]

        self.core_network = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=input_dim,
                out_features=self.core_hidden_dim,
            ),
            torch.nn.Tanh(),
            torch.nn.Linear(
                in_features=self.core_hidden_dim,
                out_features=self.core_hidden_dim,
            ),
            torch.nn.Tanh(),
        )

        for layer in self.core_network:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1.0)
                torch.nn.init.normal_(layer.bias, mean=0.0, std=1.0)

        # Stack predictor (POP = 0, NOOP = 1, PUSH = 2)
        self.stack_head = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.core_hidden_dim,
                out_features=3,
            ),
            torch.nn.Softmax(),
        )

        # Initialise final layer with zero mean and very small variance
        self.policy_head = torch.nn.Linear(
            in_features=self.core_hidden_dim,
            out_features=num_outputs,  # Discrete: action_space[0].n
        )
        torch.nn.init.normal_(self.policy_head.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.policy_head.bias, mean=0.0, std=0.01)

        # Value head
        self.value_head = torch.nn.Linear(
            in_features=self.core_hidden_dim,
            out_features=1
        )
        torch.nn.init.normal_(self.value_head.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.value_head.bias, mean=0.0, std=0.01)

        self.current_value = None

    def forward(self, inputs, state, seq_lens):

        x = inputs["obs_flat"]

        p = self.core_network(x)

        self.current_value = self.value_head(p).squeeze(1)
        logits = self.policy_head(p)

        return logits, state

    def value_function(self):
        return self.current_value  # [batches, n_agents]
