import torch


class MobilityManager:
    def __init__(self, config):
        self.config = config
        self.M = config.number_of_aps
        self.K = config.number_of_ues
        self.tpdv = dict(device=config.device_sim,
                         dtype=config.float_dtype_sim)
        self.generate_mobility_params()
        self.ue_timer = torch.ones(1, self.K, device=config.device_sim, dtype=torch.int32)
        self.ue_paused = torch.ones_like(self.ue_timer)
        self.mobility_angles = torch.zeros_like(self.ue_timer).to(**self.tpdv)
        self.mobility_speeds = torch.zeros_like(self.ue_timer).to(**self.tpdv)


    def reset(self):
        self.ue_timer = torch.ones(1, self.K, device=self.config.device_sim, dtype=torch.int32)
        self.ue_paused = torch.ones_like(self.ue_timer)
        self.mobility_angles = torch.zeros_like(self.ue_timer).to(**self.tpdv)
        self.mobility_speeds = torch.zeros_like(self.ue_timer).to(**self.tpdv)


    def generate_mobility_params(self):
        self.field_radius = None # channel maager will set this
        if self.config.ue_mobility_type == 'pedestrain':
            self.mean_speed = 3.0 / 3.6 * 0.001 # in m/ms
            self.min_speed = 1.0 / 3.6 * 0.001
            self.max_speed = 7.0 / 3.6 * 0.001
            self.mean_movement_duration = 20000
            self.mean_movement_pause = 3000

        elif self.config.ue_mobility_type == 'vehicular':
            self.mean_speed = 70.0 / 3.6 * 0.001 # in m/ms
            self.min_speed = 50.0 / 3.6 * 0.001
            self.max_speed = 120.0 / 3.6 * 0.001
            self.mean_movement_duration = 300000
            self.mean_movement_pause = 30000

        else:
            raise ValueError('Invalid mobility type')


    def generate_angles(self, indices):
        # if self.config.ue_mobility_type == 'pedestrain':
        self.mobility_angles[indices] = 2 * torch.pi * torch.rand(1, len(indices[0])).to(**self.tpdv)
        # else:
        #     raise ValueError('Invalid mobility type')


    def generate_speeds(self, indices):
        # if self.config.ue_mobility_type == 'pedestrain':
        self.mobility_speeds[indices] = torch.distributions.Exponential(1 / self.mean_speed) \
            .sample((1, len(indices[0]))).clamp(min=self.min_speed, max=self.max_speed).to(**self.tpdv)
        # else:
        #     raise ValueError('Invalid mobility type')


    def step(self, curr_x, curr_y):
        self.ue_timer -= 1
        to_pause_indices = torch.nonzero(torch.logical_and(self.ue_timer == 0, self.ue_paused == 0), 
                                         as_tuple=True)
        to_start_indices = torch.nonzero(torch.logical_and(self.ue_timer == 0, self.ue_paused == 1),
                                         as_tuple=True)

        if len(to_pause_indices[0]) > 0:
            self.ue_paused[to_pause_indices] = 1
            self.ue_timer[to_pause_indices] = torch.distributions.Exponential(1 / self.mean_movement_pause) \
                .sample([1, len(to_pause_indices[0])]).clamp(min=1).to(dtype=torch.int32, device=self.config.device_sim)
            self.mobility_speeds[to_pause_indices] = 0
            self.mobility_angles[to_pause_indices] = 0

        if len(to_start_indices[0]) > 0:
            self.ue_paused[to_start_indices] = 0
            self.ue_timer[to_start_indices] = torch.distributions.Exponential(1 / self.mean_movement_duration) \
                .sample([1, len(to_start_indices[0])]).clamp(min=1).to(dtype=torch.int32, device=self.config.device_sim)
            self.generate_speeds(to_start_indices)
            self.generate_angles(to_start_indices)

        delta_x = self.mobility_speeds * self.config.simulation_timestep * torch.cos(self.mobility_angles) 
        delta_y = self.mobility_speeds * self.config.simulation_timestep * torch.sin(self.mobility_angles) 

        tmp_x = curr_x + delta_x
        tmp_y = curr_y + delta_y

        out_of_range_indices = torch.nonzero(torch.sqrt(tmp_x ** 2 + tmp_y ** 2) > self.field_radius, as_tuple=True)
        if len(out_of_range_indices[0]) > 0:
            out_of_range_angles = torch.atan2(tmp_y[out_of_range_indices], tmp_x[out_of_range_indices])
            tmp_x[out_of_range_indices] = self.field_radius * torch.cos(out_of_range_angles)
            tmp_y[out_of_range_indices] = self.field_radius * torch.sin(out_of_range_angles)
            self.mobility_angles[out_of_range_indices] += torch.pi

        return tmp_x, tmp_y
