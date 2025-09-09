import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.entropy_models.entropy_models import GGM
from src.utils.utils import _update_registered_buffer

class BB(nn.Module):
    def __init__(self, N):
        super().__init__()
        
        self.N = int(N)
        
        self.g_a = None
        self.g_s = None
        self.h_a = None
        self.h_s = None
        
        self.means_hyper = nn.Parameter(torch.zeros(1,N,1,1))
        self.scales_hyper = nn.Parameter(torch.ones(1,N,1,1))

        self.mask_for_two_part = {}
        self.mask_for_four_part = {}
        self.mask_for_eight_part = {}
        self.mask_for_rec_s2 = {}

        self.entropy_estimation = GGM()
        
        self.init()
        
    def init(self):
        self.register_buffer("scale_table", torch.Tensor())
        self.register_buffer("quantized_cdf_z", torch.Tensor())
        self.register_buffer("cdf_length_z", torch.Tensor())
        self.register_buffer("offset_z", torch.Tensor())
        self.register_buffer("quantized_cdf_y", torch.Tensor())
        self.register_buffer("cdf_length_y", torch.Tensor())
        self.register_buffer("offset_y", torch.Tensor())
        return True
    
    def update(self, scale_table):
        self.register_buffer("scale_table", scale_table)
        quantized_cdf, cdf_length, offset = self.entropy_estimation.get_quantized_cdf(self.scales_hyper.detach().view(-1))
        self.register_buffer("quantized_cdf_z", quantized_cdf)
        self.register_buffer("cdf_length_z", cdf_length)
        self.register_buffer("offset_z", offset)
        quantized_cdf, cdf_length, offset = self.entropy_estimation.get_quantized_cdf(scale_table)
        self.register_buffer("quantized_cdf_y", quantized_cdf)
        self.register_buffer("cdf_length_y", cdf_length)
        self.register_buffer("offset_y", offset)
        return True
    
    def _update_registered_buffers(
        self,
        buffer_names,
        state_dict,
        policy="resize_if_empty",
        dtype=torch.int,
    ):
        """
        Update the registered buffers in a module according to the tensors sized
        in a state_dict.
        """
        
        valid_buffer_names = [n for n, _ in self.named_buffers()]
        
        for buffer_name in buffer_names:
            if buffer_name not in valid_buffer_names:
                raise ValueError(f'Invalid buffer name "{buffer_name}"')

        for buffer_name in buffer_names:
            _update_registered_buffer(
                self,
                buffer_name,
                f"{buffer_name}",
                state_dict,
                policy,
                dtype,
            )
    
    @staticmethod
    def build_indexes_z(size):
        dims = len(size)
        N = size[0]
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(N, 1, *size[2:])
        
    def build_indexes_conditional(self, scales):
        device = scales.device
        scale_table = self.scale_table[:-1].to(device).view(1,1,1,1,-1)
        scales_expand = scales.unsqueeze(-1)
        indexes = (scales_expand>scale_table).sum(-1)
        return indexes
    
    @staticmethod
    def compress_symbols(symbols, indexes, quantized_cdf, cdf_length, offset, encoder):
        encoder.encode_with_indexes(
            symbols.reshape(-1).int().cpu().numpy(),
            indexes.reshape(-1).int().cpu().numpy(),
            quantized_cdf,
            cdf_length,
            offset
        )
        return True
    
    @staticmethod
    def decompress_symbols(indexes, quantized_cdf, cdf_length, offset, decoder):
        values = decoder.decode_stream(
            indexes.reshape(-1).int().cpu().numpy(),
            quantized_cdf,
            cdf_length,
            offset
        )
        outputs = torch.tensor(
            values, device=indexes.device, dtype=torch.float32
        ).reshape(indexes.size())
        return outputs
    
    @staticmethod
    def ste_round(x):
        xr = torch.round(x)
        return (xr - x).detach() + x
    
    @staticmethod
    def add_noise(x):
        half = float(0.5)
        noise = torch.empty_like(x).uniform_(-half, half)
        return x + noise

    def process_with_mask(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask
        y_res = (y - means_hat) * mask
        if self.training:
            y_q = self.ste_round(y_res)
        else:
            y_q = torch.round(y_res)
        y_hat = y_q + means_hat
        return y_res, y_q, y_hat, scales_hat
    
    def get_one_channel_two_parts_mask(self, height, width, dtype, device):
        micro_mask = torch.tensor(((1, 0), (0, 1)), dtype=dtype, device=device)
        mask_0 = micro_mask.repeat(height // 2, width // 2)
        mask_0 = mask_0.unsqueeze(0).unsqueeze(0)
        mask_1 = torch.ones_like(mask_0) - mask_0
        return [mask_0, mask_1]
    
    def get_mask_two_parts(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        with torch.no_grad():
            if curr_mask_str not in self.mask_for_two_part:
                assert channel % 2 == 0
                m = torch.ones((batch, channel // 2, height, width), dtype=dtype, device=device)
                m0, m1 = self.get_one_channel_two_parts_mask(height, width, dtype, device)
                mask_0 = torch.cat((m * m0, m * m1), dim=1)
                mask_1 = torch.cat((m * m1, m * m0), dim=1)
                self.mask_for_two_part[curr_mask_str] = [mask_0, mask_1]
        return self.mask_for_two_part[curr_mask_str]
    
    def get_one_channel_four_parts_mask(self, height, width, dtype, device):
        micro_mask_0 = torch.tensor(((1, 0), (0, 0)), dtype=dtype, device=device)
        mask_0 = micro_mask_0.repeat((height + 1) // 2, (width + 1) // 2)
        mask_0 = mask_0[:height, :width]
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_0 = torch.unsqueeze(mask_0, 0)

        micro_mask_1 = torch.tensor(((0, 1), (0, 0)), dtype=dtype, device=device)
        mask_1 = micro_mask_1.repeat((height + 1) // 2, (width + 1) // 2)
        mask_1 = mask_1[:height, :width]
        mask_1 = torch.unsqueeze(mask_1, 0)
        mask_1 = torch.unsqueeze(mask_1, 0)

        micro_mask_2 = torch.tensor(((0, 0), (1, 0)), dtype=dtype, device=device)
        mask_2 = micro_mask_2.repeat((height + 1) // 2, (width + 1) // 2)
        mask_2 = mask_2[:height, :width]
        mask_2 = torch.unsqueeze(mask_2, 0)
        mask_2 = torch.unsqueeze(mask_2, 0)

        micro_mask_3 = torch.tensor(((0, 0), (0, 1)), dtype=dtype, device=device)
        mask_3 = micro_mask_3.repeat((height + 1) // 2, (width + 1) // 2)
        mask_3 = mask_3[:height, :width]
        mask_3 = torch.unsqueeze(mask_3, 0)
        mask_3 = torch.unsqueeze(mask_3, 0)

        return mask_0, mask_1, mask_2, mask_3

    def get_mask_four_parts(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        with torch.no_grad():
            if curr_mask_str not in self.mask_for_four_part:
                assert channel % 4 == 0
                m = torch.ones((batch, channel // 4, height, width), dtype=dtype, device=device)
                m0, m1, m2, m3 = self.get_one_channel_four_parts_mask(height, width, dtype, device)
                mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
                mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
                mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
                mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)
                self.mask_for_four_part[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.mask_for_four_part[curr_mask_str]
    
    def get_one_channel_eight_parts_mask(self, height, width, dtype, device):
        patten_list = [((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0)), ((0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)), \
                       ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1)), ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0)), \
                       ((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0)), ((0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0)), \
                       ((0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0)), ((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0))]
        mask_list = []
        for i in range(len(patten_list)):
            micro_mask = torch.tensor(patten_list[i], dtype=dtype, device=device)
            micro_mask = micro_mask.repeat(2, 2)
            mask = micro_mask.repeat((height + 1) // 8, (width + 1) // 8)
            mask = mask[:height, :width]
            mask = torch.unsqueeze(mask, 0)
            mask = torch.unsqueeze(mask, 0)
            mask_list.append(mask)

        return mask_list
    
    def get_one_channel_eight_parts_mask_for_s1(self, height, width, dtype, device):
        patten_list = [((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), ((0, 0, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), \
                       ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0)), \
                       ((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), ((0, 0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), \
                       ((0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), ((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0))]
        mask_list = []
        for i in range(len(patten_list)):
            micro_mask = torch.tensor(patten_list[i], dtype=dtype, device=device)
            mask = micro_mask.repeat((height + 1) // 4, (width + 1) // 4)
            mask = mask[:height, :width]
            mask = torch.unsqueeze(mask, 0)
            mask = torch.unsqueeze(mask, 0)
            mask_list.append(mask)

        return mask_list
    
    def get_one_channel_eight_parts_mask_for_s2(self, height, width, dtype, device):
        patten_list = [((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0)), ((0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)), \
                       ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1)), ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0)), \
                       ((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0)), ((0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0)), \
                       ((0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0)), ((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0))]
        mask_list = []
        for i in range(len(patten_list)):
            micro_mask = torch.tensor(patten_list[i], dtype=dtype, device=device)
            mask = micro_mask.repeat((height + 1) // 4, (width + 1) // 4)
            mask = mask[:height, :width]
            mask = torch.unsqueeze(mask, 0)
            mask = torch.unsqueeze(mask, 0)
            mask_list.append(mask)

        return mask_list

    def get_mask_eight_parts(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        with torch.no_grad():
            if curr_mask_str not in self.mask_for_eight_part:
                assert channel % 8 == 0
                mask_list = []
                m = torch.ones((batch, channel // 8, height, width), dtype=dtype, device=device)
                mask_list_one_channel = self.get_one_channel_eight_parts_mask(height, width, dtype, device)
                cat_list = [
                    [0, 2, 4, 6, 1, 3, 5, 7], 
                    [1, 3, 5, 7, 0, 2, 4, 6], 
                    [2, 4, 6, 0, 3, 5, 7, 1], 
                    [3, 5, 7, 1, 2, 4, 6, 0], 
                    [4, 6, 0, 2, 5, 7, 1, 3], 
                    [5, 7, 1, 3, 4, 6, 0, 2], 
                    [6, 0, 2, 4, 7, 1, 3, 5], 
                    [7, 1, 3, 5, 6, 0, 2, 4], 
                ]
                for i in range(8):
                    mask_list.append(torch.cat([m * mask_list_one_channel[cat_list[i][j]] for j in range(8)], dim=1))

                self.mask_for_eight_part[curr_mask_str] = mask_list
        return self.mask_for_eight_part[curr_mask_str]
    
    def get_mask_for_s1(self, batch, channel, height, width, dtype, device):
        assert channel % 8 == 0
        mask_list = []
        m = torch.ones((batch, channel // 8, height, width), dtype=dtype, device=device)
        mask_list_one_channel = self.get_one_channel_eight_parts_mask_for_s1(height, width, dtype, device)
        indices = [0, 2, 4, 6, 1, 3, 5, 7]
        for i in indices:
            mask_list.append(m * mask_list_one_channel[i])

        return mask_list
    
    def get_mask_for_rec_s2(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        with torch.no_grad():
            if curr_mask_str not in self.mask_for_rec_s2:
                assert channel % 4 == 0
                m = torch.ones((batch, channel // 4, height, width), dtype=dtype, device=device)
                m0, m1, m2, m3 = self.get_one_channel_four_parts_mask(height, width, dtype, device)
                mask_0 = m * m0
                mask_1 = m * m1
                mask_2 = m * m2
                mask_3 = m * m3
                self.mask_for_rec_s2[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.mask_for_rec_s2[curr_mask_str]
    
    def get_mask_for_s2(self, batch, channel, height, width, dtype, device):
        assert channel % 8 == 0
        mask_list = []
        m = torch.ones((batch, channel // 8, height, width), dtype=dtype, device=device)
        mask_list_one_channel = self.get_one_channel_eight_parts_mask_for_s2(height, width, dtype, device)

        indices_1 = [0,2,4,6,1,3,5,7]
        indices_2 = [1,3,5,7,0,2,4,6]
        for i in range(8):
            mask_list.append(m * mask_list_one_channel[indices_1[i]] + m * mask_list_one_channel[indices_2[i]])

        return mask_list
    
    def get_s2_hyper_with_mask(self, y, mask_list_s1, mask_list_s2, mask_list_rec_s2, batch, channel, height, width, dtype, device):
        recon_y = torch.zeros((batch, channel, height, width), device=device, dtype=dtype)
        mask = torch.cat([mask_list_s2[i] - mask_list_s1[i] for i in range(len(mask_list_s1))], dim=1)
        mask_rec_s2 = torch.cat(mask_list_rec_s2, dim=1)
        recon_y[~(mask_rec_s2.bool())] = y[mask.bool()]

        return recon_y
    
    def get_s3_hyper_with_mask(self, common_params, mask_list, batch, channel, height, width, dtype, device):
        recon_y = torch.zeros((batch, channel, height, width), device=device, dtype=dtype)
        mask = torch.cat(mask_list, dim=1)
        recon_y[~(mask.bool())] = common_params[~(mask.bool())]

        return recon_y
    
    def get_s1_s2_with_mask(self, y, mask_list, batch, channel, height, width, reduce):
        y_curr_masked_list = []
        slice = channel // reduce
        for i in range(reduce):
            y_curr = y[:, slice * i: slice * (i + 1), :, :]
            y_curr_masked = y_curr.masked_select(mask_list[i].bool()).view(batch, channel // reduce, height, width)
            y_curr_masked_list.append(y_curr_masked)

        return torch.cat(y_curr_masked_list, dim=1)
    
    def recon_for_s2_s3(self, y_curr, mask_list, batch, channel, height, width, dtype, device):
        recon_y = torch.zeros((batch, channel, height, width), device=device, dtype=dtype)
        mask = torch.cat(mask_list, dim=1)
        recon_y[mask.bool()] = y_curr.reshape(-1)
        return recon_y
    
    @staticmethod
    def combine_for_writing_s1(x):
        return sum(x.chunk(2, 1))
    
    @staticmethod
    def combine_for_writing_s2(x):
        return sum(x.chunk(4, 1))

    @staticmethod
    def combine_for_writing_s3(x):
        return sum(x.chunk(8, 1))
    
    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)
