import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        obs = ptu.from_numpy(observation)

        if self.discrete : 
            action_pt = self.forward(obs).sample()
        else:
            action_pt = self.forward(obs).rsample()
        return ptu.to_numpy(action_pt)

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            action_dist = distributions.Categorical(
                logits=self.logits_net(obs)
            )
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            
            mean = self.mean_net(obs)
            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_dist = distributions.MultivariateNormal(loc=mean, scale_tril = batch_scale_tril)
            
        return action_dist

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.

        log_probs = self.forward(obs).log_prob(actions)
        loss = -(log_probs * advantages).mean() #(Actor Loss) loss function for the polic gradient algo 
        #should not see as a loss function, it does not measure performance and its distribution depends on parameters
        #in policy gradient, the loss function going down is not a signal of good performance whatsoever. 
        #Only care about the average return.
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #skip the baseline case 

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
