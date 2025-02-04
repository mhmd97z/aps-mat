import torch
from math import log10, pi
from mat.envs.aps.lib.mobility import MobilityManager


class NlosChannelManager:
    def __init__(self, config):
        self.mobility_manager = MobilityManager(config)
        self.config_params(config)
        self.measurement_mask = torch.zeros((self.M, self.K), dtype=torch.bool)  # binary mask, torch tensor
        self.serving_mask = None  # binary mask, torch tensor
        self.measurement_aps_per_ue = None  # dictionary of torch tensors per UE
        self.large_scale_coef = None  # large-scale coefs


    def config_params(self, config):
        # setting cell radius according to morphology
        self.max_serving_ue_count = config.max_serving_ue_count
        self.max_measurment_ap_count = config.max_measurment_ap_count
        self.tpdv = dict(device=config.device_sim,
                         dtype=config.float_dtype_sim)
        self.M = config.number_of_aps
        self.K = config.number_of_ues
        self.mor = config.morphology
        self.rho_d = None  # power of the direct path   
        self.APP = float(config.ap_radiation_power)
        self.B = 20  # bandwidth (MHz)
        self.BSAntG = 0  # base station antenna gain (dB)
        self.ATAntG = 0  # access terminal antenna gain (dB)
        self.NP = -230 + 10 * log10(1.38 * (273.15 + 17)) + 30 + 10 * log10(self.B) + 60
        self.MNF = 9  # mobile noise figure in dB
        self.pLoss = 0  # building penetration loss in dB
        self.rho_d = torch.pow(torch.tensor(10.0), 
                        torch.tensor((10 * log10(self.APP) + 30 + self.BSAntG + self.ATAntG - self.NP - self.MNF) / 10)
                        ).to(**self.tpdv)
        
        if self.mor == 'urban':
            self.R = 500
            self.f = torch.tensor(2000).to(**self.tpdv)  # carrier frequency in MHz
            self.W = torch.tensor(20).to(**self.tpdv)  # street width, in meters,
            self.h = torch.tensor(20).to(**self.tpdv)  # average building height, in meters
            self.hte = torch.tensor(20).to(**self.tpdv)  # effective AP antenna height in meters
            self.hre = torch.tensor(1.5).to(**self.tpdv)  # effective mobile antenna height in meters
            self.sfstd = torch.tensor(6).to(**self.tpdv)  # slow fading standard deviation in dB
        elif self.mor == 'suburban':
            self.R = 1000
            self.f = torch.tensor(2000).to(**self.tpdv)
            self.W = torch.tensor(20).to(**self.tpdv)
            self.h = torch.tensor(10).to(**self.tpdv)
            self.hte = torch.tensor(20).to(**self.tpdv)
            self.hre = torch.tensor(1.5).to(**self.tpdv)
            self.sfstd = torch.tensor(8).to(**self.tpdv)
        elif self.mor == 'rural':
            self.R = 4000
            self.f = torch.tensor(450).to(**self.tpdv)
            self.W = torch.tensor(20).to(**self.tpdv)
            self.h = torch.tensor(5).to(**self.tpdv)
            self.hte = torch.tensor(40).to(**self.tpdv)
            self.hre = torch.tensor(1.5).to(**self.tpdv)
            self.sfstd = torch.tensor(8).to(**self.tpdv)
        self.mobility_manager.field_radius = self.R


    def generate_locations(self):
        # Generate random distances for M service antennas in disc with radius R
        d_sa = self.R * torch.sqrt(torch.rand(1, self.M)).to(**self.tpdv)
        # Generate random angles for M service antennas in disc
        theta_sa = 2 * pi * torch.rand(1, self.M).to(**self.tpdv)
        x_sa = d_sa * torch.cos(theta_sa)  # x-coordinates for the M service antennas
        y_sa = d_sa * torch.sin(theta_sa)  # y-coordinates for the M service antennas
        # Generate user coordinates in the disc with radius R
        d_m = self.R * torch.sqrt(torch.rand(1, self.K)).to(**self.tpdv)
        theta_m = 2 * pi * torch.rand(1, self.K).to(**self.tpdv)
        x_m = d_m * torch.cos(theta_m)  # x-coordinates for the K users
        y_m = d_m * torch.sin(theta_m)  # y-coordinates for the K users

        self.loc_aps = (x_sa, y_sa)
        self.loc_ues = [x_m, y_m]


    def calculate_largescale_coefs(self):
        # Compute the distance from each of the K terminals to each of the M antennas
        ddd = torch.sqrt(
            (self.loc_ues[0].repeat(self.M, 1) - self.loc_aps[0].T.repeat(1, self.K)) ** 2 +
            (self.loc_ues[1].repeat(self.M, 1) - self.loc_aps[1].T.repeat(1, self.K)) ** 2 +
            ((self.hte - self.hre)) ** 2
        ).to(**self.tpdv)

        # ITU-R propagation model
        PL = 161.04 - 7.1 * torch.log10(self.W) + 7.5 * torch.log10(self.h) - (24.37 - 3.7 * (self.h / self.hte) ** 2) * \
             torch.log10(self.hte) + (43.42 - 3.1 * torch.log10(self.hte)) * (torch.log10(ddd) - 3) + \
             20 * torch.log10(self.f / 1000) - (3.2 * (torch.log10(11.75 * self.hre)) ** 2 - 4.97)

        beta = self.sfstd * torch.randn(self.M, self.K).to(**self.tpdv) - self.pLoss  # Generate shadow fadings
        beta = torch.pow(10, ((-PL + beta) / 10))  # Linear scale
        self.large_scale_coef = torch.sqrt(beta).to(**self.tpdv)


    def assign_measurement_aps(self):
        self.measurement_mask = torch.ones_like(self.large_scale_coef, dtype=torch.bool).to(self.tpdv['device'])  # Initialize binary mask
        # self.measurement_mask = torch.zeros_like(self.large_scale_coef, dtype=torch.bool).to(self.tpdv['device'])  # Initialize binary mask
        # self.measurement_aps_per_ue = [None for _ in range(self.K)]
        # ue_count_per_ap = torch.zeros(self.M, dtype=torch.int32)  # Count of UEs assigned to each AP
        # ap_max = self.max_serving_ue_count
        # measurement_max = self.max_measurment_ap_count

        # for ue in range(self.K):
        #     ue_channels = self.large_scale_coef[:, ue]
        #     sorted_ap_indices = torch.argsort(ue_channels, descending=True)

        #     selected_aps = []
        #     for ap in sorted_ap_indices:
        #         if ue_count_per_ap[ap] < ap_max:
        #             selected_aps.append([ap])
        #             ue_count_per_ap[ap] += 1
        #             self.measurement_mask[ap, ue] = True

        #         if len(selected_aps) == measurement_max:
        #             break


    def calculate_coefs(self):
        # Considering fast-scale fading in a multiplicative manner
        small_scale_coef = torch.sqrt(torch.tensor(2.0)) / 2 \
            * (torch.randn(self.M, self.K).to(**self.tpdv)
               + torch.randn(self.M, self.K).to(**self.tpdv) * 1j)
        G = self.large_scale_coef * small_scale_coef
        small_scale_coef[~self.measurement_mask] = 1
        masked_G = self.large_scale_coef * small_scale_coef

        return G, masked_G, self.rho_d


    def reset(self):
        self.generate_locations()
        self.step()
        self.assign_measurement_aps()
        self.mobility_manager.reset()
        return self.measurement_mask


    def step(self):
        self.loc_ues[0], self.loc_ues[1] = self.mobility_manager.step(self.loc_ues[0], self.loc_ues[1])
        self.calculate_largescale_coefs()
        return self.calculate_coefs()