#TODO

import torch

def obs_to_tensor(state):
    # consider adding a 4th channel for the turn number
    board_tensor = torch.tensor(state['observation'])
    action_mask_tensor = torch.tensor(state['action_mask'].reshape(9, 9)).unsqueeze(0)
    return torch.cat((board_tensor, action_mask_tensor)).unsqueeze(0).to(torch.float32)

class Trajectory:

    def __init__(self, env, agent_1, agent_2, enable_log_prob=False):
        self.env = env
        self.agent_1 = agent_1 # first player
        self.agent_2 = agent_2 # second player
        self.enable_log_prob = enable_log_prob

    def compute(self):
        self.env.reset()

        trajectory = {
            "player_1": {"observations": [], "actions": [], "rewards": [], "log_probs": []},
            "player_2": {"observations": [], "actions": [], "rewards": [], "log_probs": []},
        }

        turn = 0
        for agent in self.env.agent_iter():
            obs, reward, termination, truncation, info = self.env.last()  # get last step info
            obs_tensor = obs_to_tensor(obs)
            if termination or truncation:
                action = None  # agent is done, no action to take
            else:
                # choose action according to agent
                if agent == "player_1":
                    if self.enable_log_prob:
                        action, log_prob = self.agent_1.pick_action(obs_tensor)
                    else:
                        action = self.agent_1.pick_action(obs_tensor)
                else:
                    if self.enable_log_prob:
                        action, log_prob = self.agent_2.pick_action(obs_tensor)
                    else:
                        action = self.agent_2.pick_action(obs_tensor)

            # Record observation, action, reward
            if not (termination or truncation):
                trajectory[agent]["observations"].append(obs['observation'])
                trajectory[agent]["actions"].append(action)
                if self.enable_log_prob:
                    trajectory[agent]["log_probs"].append(log_prob)
            if turn >= 2:
                trajectory[agent]["rewards"].append(reward)

            if action is not None:
                action = action.item()
            self.env.step(action)  # take the action (None if done)
            turn += 1
        return trajectory

# manipolate trajectory
def reinforce_update(agent, trajectory, gamma):
    rewards = trajectory['rewards']
    log_probs = trajectory['log_probs']
    log_probs_tensor = torch.cat(log_probs)
    G = torch.zeros(1)
    n = len(trajectory['observations'])
    returns = []
    for i in range(n-1, -1, -1):
        G = rewards[i] + G * gamma ** (i-n+1)
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)

    loss = - (G * log_probs_tensor).sum()
    agent.update(loss)

def reinforce(env, agent1, agent2, num_episodes, gamma=0.99, update1 = True, update2 = True):
    for _ in range(num_episodes):
        tr = Trajectory(env, agent1, agent2, True)
        trajectory = tr.compute()
        if update1:
            reinforce_update(agent1, trajectory['player_1'], gamma)
        if update2:
            reinforce_update(agent2, trajectory['player_2'], gamma)
