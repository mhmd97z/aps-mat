import sys
import gym
import torch
import arguments
from gym import spaces
from aps_gnn_gym.network_simlator import NetworkSimulator
from aps_gnn_gym.data_store import DataStore
from aps_gnn_gym.utils import get_polar, range_normalization


class ApsGnnGymBase(gym.Env):
    def __init__(self, conf):
        super(ApsGnnGymBase, self).__init__()
        self.conf = conf
        self.simulator = NetworkSimulator(conf['simulation_scenario'])
        self.history_length = self.conf['history_length']
        self.datastore = DataStore(self.history_length, ['obs'])

        num_ues = self.simulator.scenario_conf['number_of_ues']
        num_aps = self.simulator.scenario_conf['number_of_aps']
        if self.conf['use_gnn_embedding'] and \
            self.conf['simulation_scenario']['precoding_algorithm'] == 'olp':
            self.feature_length = self.conf['embedding_length'] + 1
        else:
            self.feature_length = 3

        self.action_space = spaces.MultiBinary(num_ues * num_aps)
        self.observation_space = spaces.Box(low=0, high=1, shape=(
            self.history_length, self.feature_length, num_ues * num_aps), dtype=float)

    def seed(self, seed):
        self.simulator.set_seed(seed)

    def reset(self):
        self.simulator.reset()
        state, _, mask = self.compute_state_reward()
        return state, mask

    def step(self, action):
        self.simulator.step(action)

        state, reward, mask = self.compute_state_reward()
        done = False

        return state, reward, mask, done

    def compute_state_reward(self):
        simulator_info = self.simulator.datastore.get_last_k_elements()
        serving_mask = self.simulator.serving_mask.clone().detach().flatten().to(torch.int32)

        if self.conf['use_gnn_embedding'] and \
            self.conf['simulation_scenario']['precoding_algorithm'] == 'olp':
            embedding = torch.mean(simulator_info['embedding'].clone().detach(), axis=0)
            obs = torch.cat((embedding, serving_mask.unsqueeze(0)), dim=0)
        else:
            channel_coef = torch.mean(simulator_info['channel_coef'].clone().detach(), axis=0).flatten()
            chan_magnitude, chan_phase = get_polar(channel_coef)
            obs = torch.cat((chan_magnitude.unsqueeze(0), chan_phase.unsqueeze(0), 
                             serving_mask.unsqueeze(0)), dim=0)

        self.datastore.add(obs=obs)
        state = self.datastore.get_last_k_elements()['obs']

        reward = torch.cat((simulator_info['min_sinr'].clone().detach().mean().unsqueeze(0), 
                  simulator_info['totoal_power_consumption'].clone().detach().mean().unsqueeze(0),
                  ))
        mask = self.simulator.channel_manager.measurement_mask.clone().detach().flatten().to(torch.int32)

        return state, reward, mask


class ApsGnnGymMultiAgent(gym.Env):
    def __init__(self, conf):
        super(ApsGnnGymMultiAgent, self).__init__()
        self.conf = conf
        self.simulator = NetworkSimulator(conf['simulation_scenario'])
        self.history_length = self.conf['history_length']
        self.datastore = DataStore(self.history_length, ['obs'])

        num_ues = self.simulator.scenario_conf['number_of_ues']
        num_aps = self.simulator.scenario_conf['number_of_aps']
        if self.conf['use_gnn_embedding'] and \
            self.conf['simulation_scenario']['precoding_algorithm'] == 'olp':
            self.feature_length = self.conf['embedding_length'] + 1
        else:
            self.feature_length = 3

        self.num_agents = num_ues * num_aps

        self.action_space = [spaces.Discrete(2) for _ in range(self.num_agents)]
        self.observation_space = [
            spaces.Box(low=0, high=1, 
                       shape=(self.history_length, self.feature_length), 
                       dtype=float)
            for _ in range(self.num_agents)]
        self.share_observation_space = [
            spaces.Box(low=0, high=1, 
                       shape=(self.num_agents, self.history_length, self.feature_length), 
                       dtype=float)
            for _ in range(self.num_agents)]

    def seed(self, seed):
        self.simulator.set_seed(seed)

    def reset(self):
        self.simulator.reset()
        obs, state, _, mask, info = self.compute_state_reward()

        return obs, state, mask, info

    def step(self, action):
        self.simulator.step(action)

        obs, state, reward, mask, info = self.compute_state_reward()
        done = False

        return obs, state, reward, mask, done, info

    def compute_state_reward(self):
        # state calc
        simulator_info = self.simulator.datastore.get_last_k_elements()
        serving_mask = self.simulator.serving_mask.clone().detach().flatten().to(torch.int32)

        if self.conf['use_gnn_embedding'] and \
            self.conf['simulation_scenario']['precoding_algorithm'] == 'olp':
            embedding = torch.mean(
                simulator_info['embedding'].clone().detach(), 
                axis=0)
            obs = torch.cat((embedding, serving_mask.unsqueeze(0)), dim=0)
        else:
            channel_coef = torch.mean(
                simulator_info['channel_coef'].clone().detach(), axis=0
                ).flatten()
            chan_magnitude, chan_phase = get_polar(channel_coef)
            obs = torch.cat(
                (chan_magnitude.unsqueeze(0), chan_phase.unsqueeze(0), 
                 serving_mask.unsqueeze(0)), 
                dim=0)

        # to get the history of state variables
        self.datastore.add(obs=obs)
        state = self.datastore.get_last_k_elements()['obs'].permute(2, 0, 1)
        obs = state.clone().detach()
        state = state.unsqueeze(0).repeat(self.num_agents, 1, 1, 1)

        # reward calc
        normalized_total_power_consumption = range_normalization(simulator_info['totoal_power_consumption'], 1, 5) # assumed range of power: 1, 5
        if self.conf['reward'] == 'weighted_sum':
            normalized_min_sinr = range_normalization(simulator_info['min_sinr'], -75., 25.) # assumed range of min_sinr: -75, 25
            alpha = self.conf['reward_power_consumption_coef']
            reward_ = ((1 - alpha) * normalized_min_sinr - alpha * normalized_total_power_consumption).mean()
        elif self.conf['reward'] == 'se_requirement':
            # min_sinr - threshold is expected to be > 0
            threshold = self.conf['sinr_threshold']
            constraints = simulator_info['min_sinr'] - threshold
            beta = self.conf['reward_sla_viol_coef']
            if self.conf['barrier_function'] == 'exponential':
                constraints /= 100
                se_violation_cost = torch.clip(torch.exp(-beta * constraints), max=100)
            elif self.conf['barrier_function'] == 'step':
                se_violation_cost = beta * (constraints < 0).float()
            else:
                NotImplementedError
            reward_ = (-se_violation_cost - normalized_total_power_consumption).mean()
        else:
            raise NotImplementedError
        reward = reward_.clone().detach().unsqueeze(0).unsqueeze(0).repeat(self.num_agents, 1)

        mask = self.simulator.channel_manager.measurement_mask.clone().detach() \
            .flatten().to(torch.int32).unsqueeze(1)

        info = {
            'min_sinr': simulator_info['min_sinr'].mean(),
            'totoal_power_consumption': simulator_info['totoal_power_consumption'].mean(),
            'reward': reward_.mean(),
            'se_violation_cost': se_violation_cost.mean()
        }

        return obs, state, reward, mask, info
