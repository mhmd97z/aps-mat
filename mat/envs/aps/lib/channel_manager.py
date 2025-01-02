import torch
from math import log10, pi


class NlosChannelManager:
    def __init__(self, config):
        self.config = config
        self.M = config.number_of_aps
        self.K = config.number_of_ues
        self.mor = config.morphology
        self.loc_aps = None
        self.loc_ues = None
        self.measurement_mask = torch.zeros((self.M, self.K), dtype=torch.bool)  # binary mask, torch tensor
        self.serving_mask = None  # binary mask, torch tensor
        self.measurement_aps_per_ue = None  # dictionary of torch tensors per UE
        self.large_scale_coef = None  # large-scale coefs
        self.rho_d = None
        self.tpdv = dict(device=config.device_sim,
                         dtype=config.float_dtype_sim)

    def generate_locations(self):
        # setting cell radius according to morphology
        if self.mor == 'urban':
            R = 0.5
        elif self.mor == 'suburban':
            R = 1
        elif self.mor == 'rural':
            R = 4
        # Generate random distances for M service antennas in disc with radius R
        d_sa = R * torch.sqrt(torch.rand(1, self.M)).to(**self.tpdv)
        # Generate random angles for M service antennas in disc
        theta_sa = 2 * pi * torch.rand(1, self.M).to(**self.tpdv)
        x_sa = d_sa * torch.cos(theta_sa)  # x-coordinates for the M service antennas
        y_sa = d_sa * torch.sin(theta_sa)  # y-coordinates for the M service antennas

        # Generate user coordinates in the disc with radius R
        d_m = R * torch.sqrt(torch.rand(1, self.K)).to(**self.tpdv)
        theta_m = 2 * pi * torch.rand(1, self.K).to(**self.tpdv)
        x_m = d_m * torch.cos(theta_m)  # x-coordinates for the K users
        y_m = d_m * torch.sin(theta_m)  # y-coordinates for the K users

        self.loc_aps = (x_sa, y_sa)
        self.loc_ues = (x_m, y_m)

    def calculate_largescale_coefs(self):
        if self.mor == 'urban':
            f = torch.tensor(2000).to(**self.tpdv)  # carrier frequency in MHz
            W = torch.tensor(20).to(**self.tpdv)  # street width, in meters,
            h = torch.tensor(20).to(**self.tpdv)  # average building height, in meters
            hte = torch.tensor(20).to(**self.tpdv)  # effective AP antenna height in meters
            hre = torch.tensor(1.5).to(**self.tpdv)  # effective mobile antenna height in meters
            sfstd = torch.tensor(6).to(**self.tpdv)  # slow fading standard deviation in dB
        elif self.mor == 'suburban':
            f = torch.tensor(2000).to(**self.tpdv)
            W = torch.tensor(20).to(**self.tpdv)
            h = torch.tensor(10).to(**self.tpdv)
            hte = torch.tensor(20).to(**self.tpdv)
            hre = torch.tensor(1.5).to(**self.tpdv)
            sfstd = torch.tensor(8).to(**self.tpdv)
        elif self.mor == 'rural':
            f = torch.tensor(450).to(**self.tpdv)
            W = torch.tensor(20).to(**self.tpdv)
            h = torch.tensor(5).to(**self.tpdv)
            hte = torch.tensor(40).to(**self.tpdv)
            hre = torch.tensor(1.5).to(**self.tpdv)
            sfstd = torch.tensor(8).to(**self.tpdv)

        B = 20  # bandwidth (MHz)
        APP = float(self.config.ap_radiation_power)
        BSAntG = 0  # base station antenna gain (dB)
        ATAntG = 0  # access terminal antenna gain (dB)
        NP = -230 + 10 * log10(1.38 * (273.15 + 17)) + 30 + 10 * log10(B) + 60
        MNF = 9  # mobile noise figure in dB
        pLoss = 0  # building penetration loss in dB
        self.rho_d = torch.pow(torch.tensor(10.0), 
                               torch.tensor((10 * log10(APP) + 30 + BSAntG + ATAntG - NP - MNF) / 10)
                               ).to(**self.tpdv)

        # Compute the distance from each of the K terminals to each of the M antennas
        ddd = torch.sqrt(
            (self.loc_ues[0].repeat(self.M, 1) - self.loc_aps[0].T.repeat(1, self.K)) ** 2 +
            (self.loc_ues[1].repeat(self.M, 1) - self.loc_aps[1].T.repeat(1, self.K)) ** 2 +
            ((hte - hre) / 1000) ** 2
        ).to(**self.tpdv)

        # ITU-R propagation model
        PL = 161.04 - 7.1 * torch.log10(W) + 7.5 * torch.log10(h) - (24.37 - 3.7 * (h / hte) ** 2) * \
             torch.log10(hte) + (43.42 - 3.1 * torch.log10(hte)) * (torch.log10(ddd * 1000) - 3) + \
             20 * torch.log10(f / 1000) - (3.2 * (torch.log10(11.75 * hre)) ** 2 - 4.97)

        beta = sfstd * torch.randn(self.M, self.K).to(**self.tpdv) - pLoss  # Generate shadow fadings
        beta = torch.pow(10, ((-PL + beta) / 10))  # Linear scale
        self.large_scale_coef = torch.sqrt(beta).to(**self.tpdv)

    def assign_measurement_aps(self):
        self.calculate_largescale_coefs()
        self.measurement_mask = torch.zeros_like(self.large_scale_coef, dtype=torch.bool).to(self.tpdv['device'])  # Initialize binary mask
        self.measurement_aps_per_ue = [None for _ in range(self.K)]
        ue_count_per_ap = torch.zeros(self.M, dtype=torch.int32)  # Count of UEs assigned to each AP
        ap_max = self.config.max_serving_ue_count
        measurement_max = self.config.max_measurment_ap_count

        for ue in range(self.K):
            ue_channels = self.large_scale_coef[:, ue]
            sorted_ap_indices = torch.argsort(ue_channels, descending=True)

            selected_aps = []
            for ap in sorted_ap_indices:
                if ue_count_per_ap[ap] < ap_max:
                    selected_aps.append([ap])
                    ue_count_per_ap[ap] += 1
                    self.measurement_mask[ap, ue] = True

                if len(selected_aps) == measurement_max:
                    break

    def calculate_coefs(self):
        # Considering fast-scale fading in a multiplicative manner
        small_scale_coef = torch.sqrt(torch.tensor(2.0)) / 2 \
            * (torch.randn(self.M, self.K).to(**self.tpdv)
               + torch.randn(self.M, self.K).to(**self.tpdv) * 1j)
        G = self.large_scale_coef * small_scale_coef
        small_scale_coef[~self.measurement_mask] = 1
        masked_G = self.large_scale_coef * small_scale_coef

        return G, masked_G, self.rho_d
