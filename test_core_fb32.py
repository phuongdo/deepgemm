import random
import torch
from typing import Tuple

import deep_gemm
from deep_gemm import  calc_diff, ceil_div, get_col_major_tma_aligned_tensor
import os
import sys
# import suppress_stdout_stderr
# import empty_suppress


import torch.distributed as dist
class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass
class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


# def construct(m: int, k: int, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     x = torch.randn((m, k), device='cuda', dtype=torch.float32)
#     y = torch.randn((n, k), device='cuda', dtype=torch.float32)
#     out = torch.empty((m, n), device='cuda', dtype=torch.float32)
#     ref_out = x @ y.t()
#     return x, y, out, ref_out

def bench_kineto(fn, kernel_names, num_tests: int = 30, suppress_kineto_output: bool = False,
                 trace_path: str = None, barrier_comm_profiling: bool = False, flush_l2: bool = False):
    # Conflict with Nsight Systems
    using_nsys = os.environ.get('DG_NSYS_PROFILING', False)

    # For some auto-tuning kernels with prints
    fn()

    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output and not using_nsys else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1) if not using_nsys else None
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) if not using_nsys else empty_suppress()
        with profiler:
            for i in range(2):
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                    lhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                    rhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                    lhs @ rhs
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device='cuda'))
                for _ in range(num_tests):
                    if flush_l2:
                        torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda').zero_()
                    fn()

                if not using_nsys:
                    profiler.step()

    # Return 1 if using Nsight Systems
    if using_nsys:
        return 1

    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tupled = isinstance(kernel_names, tuple)
    prof_lines = profiler.key_averages().table(sort_by='cuda_time_total', max_name_column_width=100).split('\n')
    # print(prof_lines)
    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert sum([name in line for line in prof_lines]) == 1, f'Errors of the kernel {name} in the profiling table'
        

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # Return average kernel times
    units = {'ms': 1e3, 'us': 1e6}
    kernel_times = []
    for name in kernel_names:
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                for unit, scale in units.items():
                    if unit in time_str:
                        kernel_times.append(float(time_str.replace(unit, '')) / scale)
                        break
                break
    return tuple(kernel_times) if is_tupled else kernel_times[0]



import torch
from typing import Tuple


def gemm_fp32(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """
    Perform General Matrix Multiplication (GEMM) with FP32 precision on the GPU.

    Arguments:
        lhs: A FP32 tensor of shape `[m, k]`.
        rhs: A FP32 tensor of shape `[k, n]`.

    Returns:
        A FP32 tensor of shape `[m, n]` resulting from the matrix multiplication of lhs and rhs.
    """
    # Ensure the input tensors are of type FP32
    assert lhs.dtype == torch.float32, "LHS tensor must be of type torch.float32"
    assert rhs.dtype == torch.float32, "RHS tensor must be of type torch.float32"

    # Ensure the input tensors are on the GPU
    assert lhs.is_cuda, "LHS tensor must be on the GPU"
    assert rhs.is_cuda, "RHS tensor must be on the GPU"

    # Perform matrix multiplication
    out = torch.matmul(lhs, rhs)

    return out

def construct(m: int, k: int, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct test tensors for GEMM operations.

    Arguments:
        m: Number of rows in the LHS matrix and output matrix.
        k: Number of columns in the LHS matrix and number of rows in the RHS matrix.
        n: Number of columns in the RHS matrix and output matrix.

    Returns:
        A tuple containing:
        - lhs: A FP32 tensor of shape `[m, k]` on the GPU.
        - rhs: A FP32 tensor of shape `[k, n]` on the GPU.
        - out: A FP32 tensor of shape `[m, n]` on the GPU, initialized to zeros.
        - ref_out: A FP32 tensor of shape `[m, n]` on the GPU, representing the reference output.
    """
    lhs = torch.randn(m, k, dtype=torch.float32, device='cuda')
    rhs = torch.randn(k, n, dtype=torch.float32, device='cuda')  # Changed shape to [k, n]
    out = torch.zeros(m, n, dtype=torch.float32, device='cuda')
    ref_out = torch.matmul(lhs, rhs)  # Now directly multiply lhs and rhs
    return lhs, rhs, out, ref_out

def test_fp32() -> None:
    print('Testing FP32:')
    for m in (64, 128, 4096):
        for k, n in [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
            x, y, out, ref_out = construct(m, k, n)
            out.copy_(gemm_fp32(x, y))
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

            def test_func():
                x, y, out, _ = construct(m, k, n)
                out.copy_(gemm_fp32(x, y))

            t = bench_kineto(test_func, 'gemm', suppress_kineto_output=True)
            print(f' > Performance (m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(m * k + k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')
    print()



if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)
    torch.set_float32_matmul_precision('high')  # Options: 'highest', 'high', 'medium'
    torch.cuda.set_device(1)  # Select GPU 1 (Index starts from 0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')


    test_fp32()
