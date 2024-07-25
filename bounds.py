import torch
import torch.nn.functional as F
import numpy as np
import time
from math import ceil
from itertools import product

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


############################################################################

def reshape_strided_kernel(kernel, s=1):
    '''
    :param kernel: 4d kernel
    :param s: integer stride
    :return: reshaped kernel for computation of spectral norm of strided convolution
    '''
    cout, cin, h, w = kernel.shape
    if s != 1:
        if h % s != 0 and w % s != 0:
            p = (0, s - h % s, 0, s - w % s)
            kernel = F.pad(kernel, p, 'constant', 0)
        kernel = kernel.reshape(cout, cin, ceil(h / s), s, ceil(w / s), s)
        kernel = kernel.permute(0, 1, 3, 5, 2, 4)
        kernel = kernel.reshape(cout, cin * s * s, ceil(h / s), ceil(w / s))
    return kernel


def compute_tensor_norm_einsum(kernel, pad_to=None, num_iters=100, s=1, return_time=False, times=1):
    '''
    This is implementation for CPU. It works slower on GPU, see implementation for GPU below.

    :return: upper bound on spectral norm of convolution with arbitrary padding
    upper bound is computed as spectral norm of 4-d tensor multiplied by (hw)**0.5

    computes tensor norm several times from different random initializations if times>1
    '''
    start_time = time.time()
    kernel = reshape_strided_kernel(kernel, s).to(torch.cfloat)
    cout, cin, h, w = kernel.shape
    lst = []
    for j in range(times):
        u1 = torch.randn(cout, dtype=torch.cfloat, device=device)
        u2 = torch.randn(cin, dtype=torch.cfloat, device=device)
        u3 = torch.randn(h, dtype=torch.cfloat, device=device)
        for i in range(num_iters):
            k = torch.einsum('ijkl,i->jkl', kernel, u1)
            u4 = torch.einsum('jkl,j,k->l', k, u2, u3)
            u4 = u4.conj() / torch.norm(u4)
            u3 = torch.einsum('jkl,j,l->k', k, u2, u4)
            u3 = u3.conj() / torch.norm(u3)
            u2 = torch.einsum('jkl,k,l->j', k, u3, u4)
            u2 = u2.conj() / torch.norm(u2)
            u1 = torch.einsum('ijkl,j,k,l->i', kernel, u2, u3, u4).conj()
        lst.append(torch.norm(u1))
    sigma = max(lst) * (h * w) ** 0.5
    total_time = time.time() - start_time
    if return_time:
        return sigma, total_time
    else:
        return sigma


def compute_tensor_norm(kernel, pad_to=None, num_iters=100, s=1, return_time=False):
    '''
    This implementation works fast on GPU.

    :return: upper bound on spectral norm of convolution with arbitrary padding
    upper bound is computed as spectral norm of 4-d tensor multiplied by (hw)**0.5
    '''
    kernel = reshape_strided_kernel(kernel, s).to(torch.cfloat)
    cout, cin, h, w = kernel.shape
    start_time = time.time()
    u1 = torch.randn(cout, 1, 1, 1, device=device, dtype=torch.cfloat)
    u2 = torch.randn(1, cin, 1, 1, device=device, dtype=torch.cfloat)
    u3 = torch.randn(1, 1, h, 1, device=device, dtype=torch.cfloat)
    for i in range(num_iters):
        k = kernel * u1
        k *= u3
        k = torch.sum(k, dim=[0, 2], keepdim=True)
        u4 = torch.sum(k * u2, dim=1, keepdim=True).conj()
        u4 /= torch.norm(u4)
        u2 = torch.sum(k * u4, dim=3, keepdim=True).conj()
        u2 /= torch.norm(u2)
        k = kernel * u2
        k *= u4
        k = torch.sum(k, dim=[1, 3], keepdim=True)
        u3 = torch.sum(k * u1, dim=0, keepdim=True).conj()
        u3 /= torch.norm(u3)
        u1 = torch.sum(k * u3, dim=2, keepdim=True).conj()
    sigma = torch.norm(u1) * (h * w) ** 0.5
    total_time = time.time() - start_time
    if return_time:
        return sigma, total_time
    else:
        return sigma


############################################################################

def l2_normalize(tensor, eps=1e-12):
    norm = float(torch.sqrt(torch.sum(tensor.float() * tensor.float())))
    norm = max(norm, eps)
    ans = tensor / norm
    return ans


def generate_uv(out_ch, in_ch, h, w):
    u1 = torch.randn((1, in_ch, 1, w), device=device, requires_grad=False)
    u1.data = l2_normalize(u1.data)

    u2 = torch.randn((1, in_ch, h, 1), device=device, requires_grad=False)
    u2.data = l2_normalize(u2.data)

    u3 = torch.randn((1, in_ch, h, w), device=device, requires_grad=False)
    u3.data = l2_normalize(u3.data)

    u4 = torch.randn((out_ch, 1, h, w), device=device, requires_grad=False)
    u4.data = l2_normalize(u4.data)

    v1 = torch.randn((out_ch, 1, h, 1), device=device, requires_grad=False)
    v1.data = l2_normalize(v1.data)

    v2 = torch.randn((out_ch, 1, 1, w), device=device, requires_grad=False)
    v2.data = l2_normalize(v2.data)

    v3 = torch.randn((out_ch, 1, 1, 1), device=device, requires_grad=False)
    v3.data = l2_normalize(v3.data)

    v4 = torch.randn((1, in_ch, 1, 1), device=device, requires_grad=False)
    v4.data = l2_normalize(v4.data)

    return u1, v1, u2, v2, u3, v3, u4, v4


def fantastic_four_iterations(conv_filter, u1, v1, u2, v2, u3, v3, u4, v4, num_iters=50):
    for i in range(num_iters):
        v1.data = l2_normalize((conv_filter.data * u1.data).sum((1, 3), keepdim=True).data)
        u1.data = l2_normalize((conv_filter.data * v1.data).sum((0, 2), keepdim=True).data)

        v2.data = l2_normalize((conv_filter.data * u2.data).sum((1, 2), keepdim=True).data)
        u2.data = l2_normalize((conv_filter.data * v2.data).sum((0, 3), keepdim=True).data)

        v3.data = l2_normalize((conv_filter.data * u3.data).sum((1, 2, 3), keepdim=True).data)
        u3.data = l2_normalize((conv_filter.data * v3.data).sum(0, keepdim=True).data)

        v4.data = l2_normalize((conv_filter.data * u4.data).sum((0, 2, 3), keepdim=True).data)
        u4.data = l2_normalize((conv_filter.data * v4.data).sum(1, keepdim=True).data)
    return u1, v1, u2, v2, u3, v3, u4, v4


def fantastic_four_sigma(conv_filter, u1, v1, u2, v2, u3, v3, u4, v4):
    sigma1 = torch.sum(conv_filter * u1 * v1)
    sigma2 = torch.sum(conv_filter * u2 * v2)
    sigma3 = torch.sum(conv_filter * u3 * v3)
    sigma4 = torch.sum(conv_filter * u4 * v4)
    func = torch.min
    sigma = func(func(func(sigma1, sigma2), sigma3), sigma4)
    return sigma


def compute_singla_2021(kernel, pad_to=None, num_iters=100, s=1, return_time=False):
    '''
    :return: upper bound on spectral norm of convolution
    "Fantastic Four: Differentiable Bounds on Singular Values of Convolution Layers" S.Singla & S.Feizi
    https://arxiv.org/abs/1911.10258
    https://github.com/singlasahil14/fantastic-four
    '''
    start_time = time.time()
    kernel = reshape_strided_kernel(kernel, s)  # if stride > 1, we can reshape kernel
    cout, cin, h, w = kernel.shape
    u1, v1, u2, v2, u3, v3, u4, v4 = generate_uv(*kernel.shape)
    u1, v1, u2, v2, u3, v3, u4, v4 = fantastic_four_iterations(kernel, u1, v1, u2, v2, u3, v3, u4, v4,
                                                               num_iters=num_iters)
    sigma = fantastic_four_sigma(kernel, u1, v1, u2, v2, u3, v3, u4, v4) * (h * w) ** 0.5

    total_time = time.time() - start_time
    if return_time:
        return sigma, total_time
    else:
        return sigma


############################################################################

def compute_sedghi_2019(kernel, pad_to, num_iters=None, s=1, return_time=False):
    '''
    :return: spectral norm of circular convolution with stride=1
    "The Singular Values of Convolutional Layers" H. Sedghi, V. Gupta, P. M. Long
    https://arxiv.org/abs/1805.10408
    '''
    start_time = time.time()
    # initial kernel is (cout cin h w)
    kernel = kernel.permute([2, 3, 0, 1])  # now kernel is (h w cout cin)
    transforms = torch.fft.fft2(kernel, s=pad_to, dim=[0, 1])
    singvals = torch.linalg.svdvals(transforms)
    sigma = torch.max(singvals)
    total_time = time.time() - start_time
    if return_time:
        return sigma, total_time
    else:
        return sigma


def get_ready_for_svd(kernel, pad_to, strides):
    # https://github.com/WhiteTeaDragon/practical_svd_conv
    assert len(kernel.shape) == 4  # K2 is given with shape (k, k, r1, r2)
    assert len(pad_to) == len(kernel.shape) - 2
    dim = 2
    if isinstance(strides, int):
        strides = [strides] * dim
    else:
        assert len(strides) == dim
    for i in range(dim):
        assert pad_to[i] % strides[i] == 0
        assert kernel.shape[i] <= pad_to[i]
    old_shape = kernel.shape
    kernel_tr = torch.permute(kernel, dims=[dim, dim + 1] + list(range(dim)))  # kernel is (r1, r2, k, k)
    padding_tuple = []
    for i in range(dim):
        padding_tuple.append(0)
        padding_tuple.append(pad_to[-i - 1] - kernel_tr.shape[-i - 1])
    kernel_pad = torch.nn.functional.pad(kernel_tr, tuple(padding_tuple))
    r1, r2 = kernel_pad.shape[:2]
    small_shape = []
    for i in range(dim):
        small_shape.append(pad_to[i] // strides[i])
    reshape_for_fft = torch.zeros((r1, r2, np.prod(np.array(strides))) + tuple(small_shape))
    for i in range(strides[0]):
        for j in range(strides[1]):
            reshape_for_fft[:, :, i * strides[1] + j, :, :] = kernel_pad[:, :, i:: strides[0], j:: strides[1]]
    fft_results = torch.fft.fft2(reshape_for_fft).reshape(r1, -1, *small_shape)
    # sing_vals shape is (r1, 4r2, k, k)
    transpose_for_svd = np.transpose(fft_results, axes=list(range(2, dim + 2)) + [0, 1])
    # now the shape is (k, k, r1, 4r2)
    return transpose_for_svd


def compute_senderovich_2022(kernel, pad_to, num_iters=None, s=1, return_time=False):
    '''
    :return: spectral norm of circular convolution with stride>=1
    "Towards Practical Control of Singular Values of Convolutional Layers" A. Senderovich, E. Bulatova, A. Obukhov, M. Rakhuba
    https://proceedings.neurips.cc/paper_files/paper/2022/file/46b1be2b90c6addc84efdf5d7e90eebc-Paper-Conference.pdf
    https://github.com/WhiteTeaDragon/practical_svd_conv
    '''
    start_time = time.time()
    strides = (s, s)
    # kernel is cout cin h w
    kernel = kernel.permute([2, 3, 0, 1])  # now kernel is h w cout cin
    assert kernel.shape[0] < pad_to[0]
    transpose_for_svd = get_ready_for_svd(kernel, pad_to, strides)
    singvals = torch.linalg.svdvals(transpose_for_svd)
    sigma = torch.max(singvals)
    total_time = time.time() - start_time
    if return_time:
        return sigma, total_time
    else:
        return sigma


############################################################################

def compute_ryu_2019(kernel, pad_to, num_iters=100, s=1, return_time=False):
    '''
    :return: spectral norm of zero-padded convolution with any stride
    "Plug-and-Play Methods Provably Converge with Properly Trained Denoisers"
    E. K. Ryu, J. Liu, S. Wang, X. Chen, Z. Wang, W. Yin
    https://arxiv.org/abs/1905.05406
    '''
    start_time = time.time()
    cout, cin, h, w = kernel.shape
    padding = (h // 2, w // 2)
    x = torch.normal(
        size=(1, cin,) + tuple(pad_to),
        mean=0,
        std=1,
    ).to(device)
    for i in range(num_iters):
        x_p = F.conv2d(x, kernel, stride=(s, s), padding=padding)
        x_p = x_p / torch.norm(x_p)
        x = F.conv_transpose2d(x_p, kernel, stride=(s, s), padding=padding)
        x = x / torch.norm(x)
    Wx = F.conv2d(x, kernel, stride=s, padding=padding)
    sigma = torch.sqrt(torch.sum(torch.pow(Wx, 2.0)) / torch.sum(torch.pow(x, 2.0)))
    total_time = time.time() - start_time
    if return_time:
        return sigma, total_time
    else:
        return sigma


############################################################################


def compute_araujo2021(X, num_iters=50, s=1, padding=0, device="cuda", return_time=True):
    # https://github.com/blaisedelattre/lip4conv/blob/main/bounds.py
    """
    Estimate spectral norm of convolutional layer with Araujo2021.

    From a convolutional filter, this function estimates the spectral norm of
    the convolutional layer for circular and zero padding using Araujo2021 [1]_.

    Code taken from [2]_, algo LipGrid with v2 implementation.

    Parameters
    ----------
    X : ndarray, shape (cout, cint, k, k)
        Convolutional filter.
    n_iter : int, default=50
        Number of samples.
    padding : int, default=0
        Padding used for convolutional layer.
    device : str, default="cuda"
        Device use for computation.
    return_time : bool, default True
        Return computational time.

    Returns
    -------
    sigma : float
        Largest singular value.
    time : float
        If `return_time` is True, it returns the computational time.

    References
    ----------
    .. [1] `On Lipschitz Regularization of Convolutional Layers using Toeplitz
        Matrix Theory
        <https://arxiv.org/abs/2006.08391>`_
        A Araujo, B Negrevergne, Y Chevaleyre & Jamal Atif, AAAI, 2021
    .. [2] https://github.com/MILES-PSL/Upper-Bound-Lipschitz-Convolutional-Layers/blob/master/lipschitz_bound/lipschitz_bound.py
    """
    cuda = device == "cuda"
    cout, cin, k, k2 = X.shape
    if k != k2:  # verify if kernel is square
        raise ValueError("The last 2 dim of the kernel must be equal.")
    if not k % 2 == 1:  # verify if kernel have odd shape
        raise ValueError("The dimension of the kernel must be odd.")
    device = X.device
    n_sample = num_iters
    start_time = time.time()
    # special case kernel 1x1
    if k == 1:
        ker = X.reshape(-1)
        res = torch.sqrt(torch.einsum("i,i->", ker, ker))
        res = res
        total_time = time.time() - start_time
        if return_time:
            return res, total_time
        else:
            return res
    # define search space
    x = np.linspace(0, 2 * np.pi, num=n_sample)
    w = np.array(list(product(x, x)))
    w0 = w[:, 0].reshape(-1, 1)
    w1 = w[:, 1].reshape(-1, 1)
    w0 = torch.FloatTensor(np.float32(w0))
    w1 = torch.FloatTensor(np.float32(w1))
    if cuda:
        w0 = w0.to(device)
        w1 = w1.to(device)
    p_index = torch.arange(-k + 1.0, 1.0) + padding
    H0 = p_index.repeat(k).reshape(k, k).T.reshape(-1)
    H1 = p_index.repeat(k)
    if cuda:
        H0 = H0.cuda()
        H1 = H1.cuda()
    real = torch.cos(w0 * H0 + w1 * H1).T
    imag = torch.sin(w0 * H0 + w1 * H1).T
    samples = (real, imag)
    real, imag = samples
    real = real.to(device)
    imag = imag.to(device)
    ker = X.reshape(cout * cin, -1)
    poly_real = torch.matmul(ker, real).view(cout, cin, -1)
    poly_imag = torch.matmul(ker, imag).view(cout, cin, -1)
    poly1 = torch.einsum("ijk,ijk->k", poly_real, poly_real)
    poly2 = torch.einsum("ijk,ijk->k", poly_imag, poly_imag)
    poly = poly1 + poly2
    sv_max = torch.sqrt(poly.max())
    d = (k - 1) / 2
    denom = 1 - (2 * d) / n_sample
    if denom:
        alpha = 1 / denom
    else:
        alpha = 1
    res = alpha * sv_max
    total_time = time.time() - start_time

    if return_time:
        return res, total_time
    else:
        return res


############################################################################

def compute_spectral_rescaling_conv(kernel, n_iter=1):
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least equal to 1, got {n_iter}")
    effective_iter = 0
    kkt = kernel
    log_curr_norm = 0
    for _ in range(n_iter):
        print(kkt.shape[-1])
        padding = kkt.shape[-1] - 1
        kkt_norm = kkt.norm().detach()
        kkt = kkt / kkt_norm
        log_curr_norm = 2 * (log_curr_norm + kkt_norm.log())
        kkt = F.conv2d(kkt, kkt, padding=padding)
        effective_iter += 1
    inverse_power = 2 ** (-effective_iter)
    t = torch.abs(kkt)
    t = t.sum(dim=(1, 2, 3)).pow(inverse_power)
    norm = torch.exp(log_curr_norm * inverse_power)
    t = t * norm
    return t


def compute_delattre2024(X, pad_to=None, num_iters=4, s=1, return_time=True):
    # https://github.com/blaisedelattre/lip4conv/blob/main/bounds.py
    """Estimate spectral norm of convolutional layer with Delattre2024.

    From a convolutional filter, this function estimates the spectral norm of
    the convolutional layer with zero padding using Delattre2024 [1]_.

    Parameters
    ----------
    X : ndarray, shape (cout, cint, k, k)
        Convolutional filter.
    n_iter : int, default=4
        Number of iterations.
    return_time : bool, default True
        Return computational time.

    Returns
    -------
    sigma : float
        Largest singular value.
    time : float
        If `return_time` is True, it returns the computational time.

    References
    ----------
    .. [1] `Spectral Norm of Convolutional Layers with Circular and Zero Paddings
        <https://arxiv.org/abs/2402.00240>`_
        B Delattre, Q Barthélemy & A Allauzen, arXiv, 2024
    """
    cout, cin, _, _ = X.shape
    if cin > cout:
        X = X.transpose(0, 1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # to compute time properly
    t_start = time.perf_counter()

    rescale_weights = compute_spectral_rescaling_conv(X, num_iters)
    sigma = rescale_weights.max()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    total_time = t_end - t_start

    if return_time:
        return sigma, total_time
    else:
        return sigma


def compute_delattre2023(X, pad_to=None, num_iters=4, s=1, return_time=True):
    # https://github.com/blaisedelattre/lip4conv/blob/main/bounds.py
    """Estimate spectral norm of convolutional layer with Delattre2023.
    'Efficient Bound of Lipschitz Constant for Convolutional Layers by Gram Iteration'
    B. Delattre, Q. Barthélemy, A. Araujo, A. Allauzen
    https://arxiv.org/abs/2305.16173

    From a convolutional filter, this function estimates the spectral norm of
    the convolutional layer for circular padding using [Section.3, Algo. 3] Delattre2023.

    Parameters
    ----------
    X : ndarray, shape (cout, cint, k, k)
        Convolutional filter.
    pad_to : None | int, default=None
        Size of input image. If None, pad_to is set equal to k.
    n_iter : int, default=4
        Number of iterations.
    return_time : bool, default True
        Return computational time.

    Returns
    -------
    sigma : float
        Largest singular value.
    time : float
        If `return_time` is True, it returns the computational time.
    """
    cout, cin, k, _ = X.shape
    if pad_to is None:
        pad_to = (k, k)
    if cin > cout:
        X = X.transpose(0, 1)
        cin, cout = cout, cin
    start_time = time.time()

    crossed_term = (
        torch.fft.rfft2(X, s=pad_to).reshape(cout, cin, -1).permute(2, 0, 1)
    )
    inverse_power = 1
    log_curr_norm = torch.zeros(crossed_term.shape[0]).to(device)
    for _ in range(num_iters):
        norm_crossed_term = crossed_term.norm(dim=(1, 2))
        crossed_term /= norm_crossed_term.reshape(-1, 1, 1)
        log_curr_norm = 2 * log_curr_norm + norm_crossed_term.log()
        crossed_term = torch.bmm(crossed_term.conj().transpose(1, 2), crossed_term)
        inverse_power /= 2
    sigma = (
            crossed_term.norm(dim=(1, 2)).pow(inverse_power)
            * ((2 * inverse_power * log_curr_norm).exp())
    ).max()
    total_time = time.time() - start_time

    if return_time:
        return sigma, total_time
    else:
        return sigma
