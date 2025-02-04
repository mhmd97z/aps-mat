import torch
from mat.envs.aps.lib.channel_manager import NlosChannelManager
from mat.envs.aps.lib.utils import set_random_seed
from mat.envs.aps.lib.data_store import DataStore
import logging

logger = logging.getLogger(__name__)


class NetworkSimulator:
    def __init__(self, conf):
        self.scenario_conf = conf
        self.number_of_aps = self.scenario_conf.number_of_aps
        self.number_of_ues = self.scenario_conf.number_of_ues
        self.seed = self.scenario_conf.seed
        self.step_length = self.scenario_conf.step_length
        self.tpdv = dict(device=conf.device_sim, type=conf.float_dtype_sim)

        self.serving_mask = torch.zeros((self.number_of_aps, self.number_of_ues), 
                                        dtype=torch.bool, device=conf.device_sim)

        self.datastore = DataStore(self.step_length,
                                            ['channel_coef', 'power_coef',
                                             'sinr', 'embedding',
                                             'totoal_power_consumption',
                                             'graph'])
        if self.scenario_conf.precoding_algorithm == "olp":
            from mat.envs.aps.lib.power_control import OlpGnnPowerControl
            self.power_control = OlpGnnPowerControl(self.scenario_conf)
        elif self.scenario_conf.precoding_algorithm == "mrt" or self.scenario_conf.precoding_algorithm == "optimal":
            from mat.envs.aps.lib.power_control import MrtPowerControl
            self.power_control = MrtPowerControl(self.scenario_conf)
        else:
            raise NotImplementedError()

        self.channel_manager = NlosChannelManager(self.scenario_conf)


    def set_seed(self, seed):
        self.seed = seed


    def reset(self):
        self.seed += 1
        set_random_seed(self.seed)
        self.measurement_mask = self.channel_manager.reset()
        self.step(self.measurement_mask)


    def step(self, connection_choices):
        self.serving_mask = connection_choices.reshape(
            (self.number_of_aps,
             self.number_of_ues)).to(self.tpdv['device'])
        self.serving_mask *= self.measurement_mask

        for _ in range(self.step_length):
            # simulator should know everything!! => calculating channel coef with full obsevability
            G, masked_G, rho_d = self.channel_manager.step()  # adding small-scale measurements
            if self.scenario_conf.precoding_algorithm == "optimal":
                _, allocated_power = self.power_control.get_optimal_sinr(G, rho_d) # allocating power
                embedding, graph = None, None
                allocated_power = torch.from_numpy(allocated_power).to(G)
            else:
                allocated_power, embedding, graph = self.power_control.get_power_coef(G, rho_d, return_graph=True)
            # to simulate aps, we set the non-activated power coef to zero
            masked_allocated_power = allocated_power.clone().detach() * self.serving_mask
            # calc total power consumption
            totoal_power_consumption = self.power_control.get_power_consumption(masked_allocated_power)
            # calc sinr with full channel info and the maked allocated power
            sinr = self.power_control.calcualte_sinr(G, rho_d, masked_allocated_power)

            # store the info
            self.datastore.add(channel_coef=masked_G, power_coef=masked_allocated_power, 
                               embedding=embedding, sinr=sinr,
                               totoal_power_consumption=totoal_power_consumption,
                               graph=graph)   # add to the data store

        # self.channel_manager.assign_measurement_aps()

