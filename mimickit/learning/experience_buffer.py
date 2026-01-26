import torch

class ExperienceBuffer():
    def __init__(self, buffer_length, batch_size, device):
        self._buffer_length = buffer_length
        self._batch_size = batch_size
        self._device = device

        self._buffer_head = 0
        self._total_samples = 0

        self._buffers = dict()
        self._flat_buffers = dict()
        self._sample_buf = torch.randperm(self.get_capacity(), device=self._device, dtype=torch.long)
        self._sample_buf_head = 0
        self._reset_sample_buf()
        return

    def add_buffer(self, name, data_shape, dtype):
        assert(name not in self._buffers)

        buffer_shape = [self._buffer_length, self._batch_size] + list(data_shape)
        buffer = torch.zeros(buffer_shape, dtype=dtype, device=self._device)
        self._buffers[name] = buffer

        flat_shape = [buffer_shape[0] * buffer_shape[1]] + list(data_shape)
        self._flat_buffers[name] = buffer.view(flat_shape)
        return

    def reset(self):
        self._buffer_head = 0
        self._reset_sample_buf()
        return

    def clear(self):
        self.reset()
        self._total_samples = 0
        return

    def inc(self):
        self._buffer_head = (self._buffer_head + 1) % self._buffer_length
        self._total_samples += self._batch_size
        return

    def get_total_samples(self):
        return self._total_samples

    def get_capacity(self):
        return self._buffer_length * self._batch_size

    def get_sample_count(self):
        sample_count = min(self._total_samples, self.get_capacity())
        return sample_count

    def is_full(self):
        return self._total_samples >= self.get_capacity()

    def record(self, name, data):
        assert(data.shape[0] == self._batch_size)

        sample_count = self.get_sample_count()
        if (sample_count == 0 and name not in self._buffers):
            self.add_buffer(name, data.shape[1:], data.dtype)

        data_buf = self._buffers[name]
        data_buf[self._buffer_head] = data
        return

    def get_data(self, name):
        return self._buffers[name]

    def get_data_flat(self, name):
        return self._flat_buffers[name]
    
    def set_data(self, name, data):
        assert(data.shape[0] == self._buffer_length)
        assert(data.shape[1] == self._batch_size)
        
        if (name not in self._buffers):
            self.add_buffer(name, data.shape[2:], data.dtype)
        
        data_buf = self.get_data(name)
        data_buf[:] = data
        return
    
    def set_data_flat(self, name, data):
        assert(data.shape[0] == self._buffer_length * self._batch_size)
        
        if (name not in self._buffers):
            self.add_buffer(name, data.shape[1:], data.dtype)

        data_buf = self.get_data_flat(name)
        data_buf[:] = data
        return

    def sample(self, n):
        output = dict()
        rand_idx = self._sample_rand_idx(n)

        for key, data in self._flat_buffers.items():
            batch_data = data[rand_idx]
            output[key] = batch_data

        return output
    
    def push(self, data_dict):
        if (len(self._buffers) == 0):
            for key, data in data_dict.items():
                self.add_buffer(name=key, data_shape=data.shape[2:], dtype=data.dtype)

        n = next(iter(data_dict.values())).shape[0]
        assert(n <= self._buffer_length)

        for key, curr_buf in self._buffers.items():
            curr_data = data_dict[key]
            curr_n = curr_data.shape[0]
            curr_batch_size = curr_data.shape[1]
            assert(n == curr_n)
            assert(curr_batch_size == self._batch_size)

            store_n = min(curr_n, self._buffer_length - self._buffer_head)
            curr_buf[self._buffer_head:(self._buffer_head + store_n)] = curr_data[:store_n]    
        
            remainder = n - store_n
            if (remainder > 0):
                curr_buf[0:remainder] = curr_data[store_n:]  

        self._buffer_head = (self._buffer_head + n) % self._buffer_length
        self._total_samples += n
        return


    def _reset_sample_buf(self):
        self._sample_buf[:] = torch.randperm(self.get_capacity(), device=self._device,
                                             dtype=torch.long)
        self._sample_buf_head = 0
        return

    def _sample_rand_idx(self, n):
        buffer_len = self._sample_buf.shape[0]
        assert(n <= buffer_len)

        if (self._sample_buf_head + n <= buffer_len):
            rand_idx = self._sample_buf[self._sample_buf_head:self._sample_buf_head + n]
            self._sample_buf_head += n
            
        else:
            rand_idx0 = self._sample_buf[self._sample_buf_head:]
            remainder = n - (buffer_len - self._sample_buf_head)

            self._reset_sample_buf()
            rand_idx1 = self._sample_buf[:remainder]
            rand_idx = torch.cat([rand_idx0, rand_idx1], dim=0)

            self._sample_buf_head = remainder

        sample_count = self.get_sample_count()
        rand_idx = torch.remainder(rand_idx, sample_count)
        return rand_idx


class PrioritizedExperienceBuffer(ExperienceBuffer):
    def __init__(self, buffer_length, batch_size, device, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        super().__init__(buffer_length, batch_size, device)
        self._alpha = alpha  # 优先级指数
        self._beta = beta  # 重要性采样权重指数
        self._beta_increment = beta_increment  # beta的增量
        self._epsilon = epsilon  # 防止优先级为0
        
        # 初始化优先级缓冲区
        self._priorities = torch.zeros(self.get_capacity(), dtype=torch.float32, device=self._device)
        self._max_priority = 1.0  # 初始最大优先级
        return
    
    def reset(self):
        super().reset()
        self._priorities.zero_()
        self._max_priority = 1.0
        return
    
    def clear(self):
        super().clear()
        self._priorities.zero_()
        self._max_priority = 1.0
        return
    
    def record(self, name, data):
        super().record(name, data)
        
        # 为新记录的样本设置优先级
        if name in ['rewards', 'r']:
            # 计算误差作为优先级的基础
            error = torch.abs(data).squeeze()
            start_idx = self._buffer_head * self._batch_size
            end_idx = start_idx + self._batch_size
            
            # 更新优先级
            self._priorities[start_idx:end_idx] = (error + self._epsilon) ** self._alpha
            self._max_priority = max(self._max_priority, error.max().item() + self._epsilon)
        return
    
    def sample(self, n):
        output = dict()
        
        # 获取当前样本数量
        sample_count = self.get_sample_count()
        
        # 计算采样概率（在CPU上进行，避免CUDA异步错误）
        priorities = self._priorities[:sample_count].cpu()
        
        try:
            # 添加小的epsilon值确保所有优先级都为正
            priorities = priorities + self._epsilon
            
            # 确保所有优先级都为正
            priorities = torch.clamp(priorities, min=self._epsilon)
            
            # 计算采样概率
            total_priority = priorities.sum()
            if total_priority <= 0:
                # 如果总优先级为0，使用均匀分布
                sampling_probabilities = torch.ones_like(priorities) / len(priorities)
            else:
                sampling_probabilities = priorities / total_priority
            
            # 确保概率都在有效范围内
            sampling_probabilities = torch.clamp(sampling_probabilities, min=0.0, max=1.0)
            
            # 重新归一化概率
            total_prob = sampling_probabilities.sum()
            if total_prob > 0:
                sampling_probabilities = sampling_probabilities / total_prob
            else:
                # 如果总概率为0，使用均匀分布
                sampling_probabilities = torch.ones_like(sampling_probabilities) / len(sampling_probabilities)
            
            # 采样索引（在CPU上进行）
            rand_idx = torch.multinomial(sampling_probabilities, n, replacement=True)
            
            # 计算重要性采样权重
            weights = (sample_count * sampling_probabilities[rand_idx]) ** (-self._beta)
            weights = weights / weights.max()
            
            # 移回原始设备
            rand_idx = rand_idx.to(self._device)
            weights = weights.to(self._device)
        except Exception as e:
            # 如果出现任何错误，使用均匀采样
            print(f"采样出错，使用均匀采样: {e}")
            rand_idx = torch.randint(0, sample_count, (n,), device=self._device)
            weights = torch.ones(n, device=self._device)
        
        # 增加beta值
        self._beta = min(1.0, self._beta + self._beta_increment)
        
        # 收集样本数据
        for key, data in self._flat_buffers.items():
            batch_data = data[rand_idx]
            output[key] = batch_data
        
        # 添加重要性采样权重
        output['weights'] = weights
        output['indices'] = rand_idx
        
        return output
    
    def update_priorities(self, indices, errors):
        # 根据新的误差更新优先级
        priorities = (errors + self._epsilon) ** self._alpha
        self._priorities[indices] = priorities
        self._max_priority = max(self._max_priority, priorities.max().item())
        return
    
    def get_beta(self):
        return self._beta
    
    def set_alpha(self, alpha):
        self._alpha = alpha
        return
    
    def set_beta(self, beta):
        self._beta = beta
        return