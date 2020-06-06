import os

import torch

from diskcache.persistent import Deque


class CudaRepository(object):

    _avaliable_devices: Deque = Deque(directory=os.path.join("output", "cuda_devices"))

    @classmethod
    def fill(cls):
        CudaRepository._avaliable_devices.clear()
        CudaRepository._avaliable_devices.extend(
            [i for i in range(torch.cuda.device_count())]
        )

    @classmethod
    def get_avaliable_device(self) -> int:
        try:
            return CudaRepository._avaliable_devices.pop()
        except IndexError:
            return 0

    @classmethod
    def put_available_device(self, device: int):
        CudaRepository._avaliable_devices.append(device)
