import numpy as np
import torch as th
from stable_baselines3 import DQN

class MaskedDQN(DQN):
    def predict(self, observation, state=None, episode_start=None, deterministic=True, action_masks=None):
        if action_masks is None:
            return super().predict(observation, state, episode_start, deterministic)
        
        # Get q values
        obs_tensor = th.tensor(observation).float().unsqueeze(0).to(self.device)
        with th.no_grad():
            q_values =self.q_net(obs_tensor).cpu().numpy()[0]

            # mask invalid actions with large negative value
            masked_q = np.where(action_masks == 1, q_values, -1e8)

        if deterministic:
            action = np.argmax(masked_q)
        else:
            valid_q = masked_q[action_masks == 1]
            temperature = 1.0
            probs = np.exp(valid_q / temperature) / np.sum(np.exp(valid_q / temperature))
            valid_indices = np.where(action_masks == 1)[0]
            action = np.random.choice(valid_indices, p=probs)

        return action, state