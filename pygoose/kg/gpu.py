import os


def _cuda_get_gpu_spec_string(gpu_ids=None):
    """
    Build a GPU id string to be used for CUDA_VISIBLE_DEVICES.
    """

    if gpu_ids is None:
        return ''

    if isinstance(gpu_ids, list):
        return ','.join(str(gpu_id) for gpu_id in gpu_ids)

    if isinstance(gpu_ids, int):
        return str(gpu_ids)

    return gpu_ids


def cuda_use_gpus(gpu_ids):
    """
    Restrict visible GPU devices only to the specified device IDs.
    The order of the IDs is determined by PCI bus.

    Examples:
        Use only the first GPU:
            `cuda_use_gpus(0)`

        Use only the second GPU:
            `cuda_use_gpus(1)`

        Use the first three GPUs:
            `cuda_use_gpus([0, 1, 2])`

    Args:
        gpu_ids: The list of GPU ids to make visible to the current application.
    """

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = _cuda_get_gpu_spec_string(gpu_ids)


def cuda_disable_gpus():
    """
    Hide all GPUs from CUDA.
    """

    cuda_use_gpus(None)
