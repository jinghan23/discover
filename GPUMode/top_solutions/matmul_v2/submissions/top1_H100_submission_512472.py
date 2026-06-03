import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    a, b, c = data
    return a @ b
