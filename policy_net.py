"""
Policy network
"""

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch


class PolicyNetwork(TorchModelV2, torch.nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name, *args, **kwargs):

        # Call super class constructors
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_config, name)
        torch.nn.Module.__init__(self)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Process keyword arguments
        self.core_hidden_dim = kwargs.get("core_hidden_dim")
        self.head_hidden_dim = kwargs.get("head_hidden_dim")

        # input_dim = observation_space.shape[0]
        input_dim = observation_space.shape[0] + self.core_hidden_dim

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

        # Make the stack. Assume sgd_minibatch_size is 256
        # 10k train batch size * 256 sgd_minibatch_size * 256 core_hidden_dim = 0.7 GiB stack
        # This is feasible. Let network think stack is large enough to be effectively infinite.
        # TODO: Rethink this in terms of episode length
        print("Constructing stack")
        self.stack = torch.empty((256, 100, self.core_hidden_dim))
        self.stack_idx = torch.zeros(256).int()
        # Set bottom of stack to all zeroes
        self.stack[self.stack_idx] = torch.zeros(self.core_hidden_dim)
        self.stack_idx += 1
        print("Finished constructing stack")

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

        # Detect anomalies for gradient computation
        torch.autograd.set_detect_anomaly(True)

    def forward(self, inputs, state, seq_lens):

        # FIXME: Not using spatial information
        x = inputs["obs_flat"]

        batch_size = x.shape[0]

        p = self.core_network(
            torch.cat(
                (
                    x,
                    self.stack[torch.arange(batch_size), self.stack_idx[:batch_size], :]
                ),
                dim=1,
            )
        )

        # stack_ops = torch.nn.functional.gumbel_softmax(
        #     self.stack_head(p)
        # )
        # breakpoint()
        stack_ops = torch.argmax(self.stack_head(p), dim=1)

        # Write current state to the next stack element
        self.stack[torch.arange(batch_size), self.stack_idx[:batch_size] + 1] = p.clone()

        # Take predicted stack action
        self.stack_idx[:batch_size] = torch.maximum(
            self.stack_idx[:batch_size] + stack_ops - 1,
            torch.zeros(batch_size).int()
        )

        self.current_value = self.value_head(p).squeeze(1)
        logits = self.policy_head(p)

        return logits, state

    def value_function(self):
        return self.current_value  # [batches, n_agents]
