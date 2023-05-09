import torch
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
class StackNetwork(TorchRNN, torch.nn.Module):
    def __init__(
            self,
            observation_space,
            action_space,
            num_outputs,
            model_config,
            name,
            fc_size=64,
            lstm_state_size=256,
            **kwargs
    ):
        torch.nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Process keyword arguments
        self.core_hidden_dim = kwargs.get("core_hidden_dim")
        self.head_hidden_dim = kwargs.get("head_hidden_dim")
        self.num_outputs = num_outputs

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

        self.stack_size = 201


    def forward_rnn(
        self, inputs, state, seq_lens
    ):

        x = inputs  # [batches, time, obs size]

        stack, stack_ptrs = state  # Stack -> [batches, 200 (maze), elem size]
        stack_ptrs = stack_ptrs.long()  # Ptrs -> [batches, 1]

        batch_size = x.shape[0]
        max_time_steps = stack.shape[1]
        time_steps = x.shape[1]

        logitss = torch.empty((batch_size, time_steps, self.num_outputs), device=self.device)
        values = torch.empty((batch_size * time_steps), device=self.device)
        for time_step in range(time_steps):
            breakpoint()
            p = self.core_network(
                torch.cat(
                    (
                        x[:, time_step, :],  # [32, dsad]
                        stack[torch.arange(batch_size), stack_ptrs[:batch_size].squeeze(-1), :],
                    ),
                    dim=1,
                )
            )

            stack_ops = torch.argmax(self.stack_head(p), dim=1)
            new_stacks = []

            for i in range(batch_size):

                if stack_ops[i] == 2:
                    stack_el = torch.cat(
                        (
                            stack[i, :stack_ptrs[i].squeeze(), :],  # [A, elem size]
                            p[i].unsqueeze(0),  # [1, elem size]
                            torch.empty((max_time_steps - stack_ptrs[i] - 1, self.core_hidden_dim), device=self.device)  # [200 - A - 2, elem size]
                        )
                    )
                else:
                    stack_el = stack[i]

                new_stacks.append(stack_el)

            stack_ptrs = torch.maximum(
                stack_ptrs[:batch_size].squeeze(-1) + stack_ops - 1,
                torch.zeros(batch_size, dtype=torch.long)
            ).unsqueeze(-1)

            stack = torch.stack(new_stacks)

            current_value = self.value_head(p).squeeze(1)
            logits = self.policy_head(p)
            logitss[:, time_step, :] = logits
            values[batch_size * time_step: batch_size * (time_step + 1)] = current_value

        self.current_value = values

        state = [
            stack,
            stack_ptrs
        ]

        # Logits -> [batch, time steps, actions]
        # Value -> [batches * time steps]
        return logitss, state

    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.

        h = [
            # Stack
            torch.zeros((self.stack_size, self.core_hidden_dim), device=self.device),

            # Stack pointers
            torch.zeros(1, dtype=torch.long, device=self.device)
        ]

        return h

    def value_function(self):
        return self.current_value  # [batches, n_agents]


