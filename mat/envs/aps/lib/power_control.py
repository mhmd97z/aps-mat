import yaml
import torch
import numpy as np
from torch_geometric.data import HeteroData
from mat.envs.aps.lib.gnn_olp.gnn import FastGNNLinearPrecodingLightning
from mat.envs.aps.lib.utils import opti_OLP, clip_abs


class PowerControl:
    def __init__(self, conf):
        self.conf = conf
        self.tpdv = dict(
            device=self.conf.device_sim,
            dtype=self.conf.float_dtype_sim)

    def get_power_coef(self, G, rho_d):
        raise NotImplementedError("Subclasses must implement this method")

    def calcualte_sinr(self, G, rho_d, P):
        recv_power = G.T @ P # row i, col j: recv power at ue i, intended for ue j
        intened_power = torch.diag(recv_power)
        interfernce_power = recv_power.fill_diagonal_(0)
        numerator = rho_d * torch.abs(intened_power)**2
        denominator = 1 + rho_d * torch.sum(torch.abs(interfernce_power)**2, axis=1)
        sinr = numerator / denominator
        # to avoid -inf values:
        sinr[sinr == 0] = 1e-20
        
        if self.conf.if_sinr_in_db:
            return 10*torch.log10(sinr)
        else:
            return sinr

    def get_power_consumption(self, allocated_power):
        return torch.sum(torch.norm(allocated_power, p=2, dim=1) ** 2) \
            * self.conf.ap_radiation_power

    def get_optimal_sinr(self, G, rho_d):
        low, up, eps = 0, 10**6, 0.01
        M, K = G.shape
        G = G.cpu().numpy()
        rho_d = rho_d.cpu().numpy()

        G_inv = np.linalg.inv((np.conjugate(G).T).dot(G))
        G_dague = np.conjugate(G).dot(G_inv.T)
        P_G = np.eye(M) - G_dague.dot(G.T)

        U_opt = np.zeros((M, K))
        A_opt = np.zeros((K, K))
        U_test = np.zeros((M, K))
        A_test = np.zeros((K, K))
        lowb = min(low, up)
        upb = max(low, up)
        ite = 0
        best_SINR = 0.0
        while abs(lowb-upb) > eps:
            ite += 1
            tSINR = (lowb+upb) / 2
            try:
                prob, A_test, U_test = opti_OLP(
                    tSINR, G_dague, P_G, rho_d, M, K)
                is_feasible = prob.value < np.inf
            except:
                is_feasible = False

            if is_feasible:
                lowb = tSINR
                A_opt, U_opt = A_test.value, U_test.value
                best_SINR = tSINR
            else:
                upb = tSINR

        Delta_opt = G_dague @ A_opt + P_G @ U_opt
        best_SINR = 10*np.log10(best_SINR)

        return best_SINR, Delta_opt


class OlpGnnPowerControl(PowerControl):
    def __init__(self, conf):
        super().__init__(conf)
        with open(self.conf.data_normalization_config, 'r') as config_file:
            self.normalization_dict = yaml.safe_load(config_file)

        self.graph = None
        self.graph_shape = None

        self.load_model()

    def load_model(self):
        self.model = FastGNNLinearPrecodingLightning.load_from_checkpoint(
            self.conf.power_control_saved_model
        )
        self.model = self.model.eval()
        self.model = self.model.to(**self.tpdv)

    def graph_generation(self, n_aps, n_ues):
        same_ap_edges = []
        same_ue_edges = []  # edges id from 0 to n_ues*n_aps-1
        # UE type edges
        # for k in range(n_ues):
        #     for m1 in range(n_aps):
        #         for m2 in range(m1 + 1, n_aps):
        #             same_ue_edges.append([k * n_aps + m1, k * n_aps + m2])
        #             # reverse to make graph unoriented
        #             same_ue_edges.append([k * n_aps + m2, k * n_aps + m1])
        for cntr_1 in range(n_ues * n_aps):
            for cntr_2 in range(n_ues * n_aps):
                if cntr_1 == cntr_2:
                    continue
                if cntr_1 % n_ues == cntr_2 % n_ues:
                    same_ue_edges.append((cntr_1, cntr_2))
                elif int(cntr_1 / n_ues) == int(cntr_2 / n_ues):
                    same_ap_edges.append((cntr_1, cntr_2))
                else:
                    pass

        same_ue_edges = torch.tensor(same_ue_edges).t().contiguous().to(self.tpdv['device'])
        same_ap_edges = torch.tensor(same_ap_edges).t().contiguous().to(self.tpdv['device'])
        # # AP type edges
        # for m in range(n_aps):
        #     for k1 in range(n_ues):
        #         for k2 in range(k1 + 1, n_ues):
        #             same_ap_edges.append([k1 * n_aps + m, k2 * n_aps + m])
        #             # reverse to make graph unoriented
        #             same_ap_edges.append([k2 * n_aps + m, k1 * n_aps + m])

        data = HeteroData()
        data['channel'].x = None
        data['channel', 'same_ue', 'channel'].edge_index = same_ue_edges
        data['channel', 'same_ap', 'channel'].edge_index = same_ap_edges
        return data

    def get_power_coef(self, G, rho_d, return_graph=False):
        # pre-process
        number_of_aps, number_of_ues = G.shape
        if self.graph_shape != (number_of_aps, number_of_ues):
            self.graph_shape = (number_of_aps, number_of_ues)
            self.graph = self.graph_generation(number_of_aps, number_of_ues)

        G = clip_abs(G)
        G_T = G.T
        G_conj = torch.conj(G)
        G_inv = torch.inverse(G_conj.T @ G)
        G_dague = G_conj @ G_inv.T
        x = torch.reshape(G_T, (-1, 1))
        x1 = torch.reshape(G_dague.T, (-1, 1))
        x = torch.cat((torch.log2(torch.abs(x)), x.angle(),
                       torch.log2(torch.abs(x1)+1), x1.angle()), 1)
        x_mean = torch.tensor(self.normalization_dict['x_mean']).to(**self.tpdv)
        x_std = torch.tensor(self.normalization_dict['x_std']).to(**self.tpdv)
        x = (x - x_mean) / x_std

        self.graph['channel'].x = x.to(**self.tpdv)
        self.graph['channel'].input_mean = torch.reshape(x_mean, (1, 4)).to(**self.tpdv)
        self.graph['channel'].input_std = torch.reshape(x_std, (1, 4)).to(**self.tpdv)
        self.graph['channel'].n_ues = number_of_ues
        self.graph['channel'].n_aps = number_of_aps
        self.graph['channel'].num_graph_node = number_of_ues * number_of_aps
        self.graph['channel'].rho_d = rho_d.to(**self.tpdv)

        with torch.no_grad():
            y, penultimate = self.model(self.graph)
            y, penultimate = y.to(**self.tpdv), penultimate.to(**self.tpdv)

        # post-process
        output_mean = torch.tensor(self.normalization_dict['y_mean']).to(**self.tpdv)
        output_std = torch.tensor(self.normalization_dict['y_std']).to(**self.tpdv)
        y = y * output_std + output_mean
        y = torch.polar(torch.pow(2, y[:, [0, 2, 4]]),
                        y[:, [1, 3, 5]])

        y1 = y[:, 0].view(number_of_ues, number_of_aps).T
        y2 = y[:, 1].view(number_of_ues, number_of_aps).T
        y3 = y[:, 2].view(number_of_ues, number_of_aps).T - 1e-20

        if self.tpdv['dtype'] == torch.float32:
            complex_type = torch.complex64
        if self.tpdv['dtype'] == torch.float64:
            complex_type = torch.complex128
        A1 = torch.matmul(G_T, y1).real.to(complex_type)  # (n_ue * n_ap) * (n_ap * n_ue)
        A1 = torch.diag(torch.diag(A1))  # (n_ue * n_ue)
        y1 = torch.matmul(G_dague, A1)  # (n_ap * n_ue) * (n_ue * n_ue)
        A2 = torch.matmul(G_T, y2)
        y2 = torch.matmul(G_dague, A2 - torch.diag(torch.diag(A2)))
        power_coef = y1 + y2 + y3

        if return_graph:
            return power_coef, penultimate.view(-1, number_of_aps * number_of_ues), self.graph.clone()
        else:
            return power_coef, penultimate.view(-1, number_of_aps * number_of_ues)


class MrtPowerControl(PowerControl):
    def __init__(self, conf):
        super().__init__(conf)

    def get_power_coef(self, G, rho_d):
        # assumes full obsevability
        number_of_aps, number_of_ues = G.shape
        power_budget = torch.ones_like(G).to(**self.tpdv) \
            * torch.sqrt(torch.tensor(1 / number_of_ues)).to(**self.tpdv)
        power_coef = torch.conj(G) / torch.abs(G) * power_budget

        return power_coef, None
