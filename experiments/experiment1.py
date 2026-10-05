import csv
import os
import re
import struct
import zlib
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / 'results'
ACT_FL = 31 - np.ceil(np.log2([0.8617, 1.217, 2.855, 2.65, 2.749, 43.01]))


def read_model():
    # The MATLAB network is an MCOS object; extract its numeric MAT matrices.
    def tag(data, offset):
        kind, size = struct.unpack_from('<II', data, offset)
        if kind >> 16:
            size = kind >> 16
            return kind & 65535, data[offset+4:offset+4+size], offset+8
        return kind, data[offset+8:offset+8+size], offset+8+((size+7)//8)*8

    raw = (ROOT / 'HW2_model.mat').read_bytes()[128:]
    arrays = []
    offset = 0
    while offset + 8 <= len(raw):
        kind, size = struct.unpack_from('<II', raw, offset)
        data = raw[offset+8:offset+8+size]
        if kind == 15:
            data = zlib.decompress(data)
        offset += 8 + size
        for match in re.finditer(b'\x0e\x00\x00\x00....\x06\x00\x00\x00\x08\x00\x00\x00', data, re.DOTALL):
            try:
                _, flags, p = tag(data, match.start()+8)
                if struct.unpack_from('<I', flags)[0] & 255 not in (6, 7):
                    continue
                _, dims, p = tag(data, p)
                shape = struct.unpack('<'+'i'*(len(dims)//4), dims)
                _, name, p = tag(data, p)
                dtype, values, p = tag(data, p)
                if dtype in (7, 9):
                    array = np.frombuffer(values, dtype={7: '<f4', 9: '<f8'}[dtype])
                    arrays.append(array.reshape(shape, order='F').astype(float))
            except (ValueError, struct.error):
                continue

    def get(shape):
        matches = [x for x in arrays if x.shape == shape]
        assert len(matches) == 1, (shape, len(matches))
        return matches[0]

    weights = [get((3, 3, 3, 64)), get((3, 3, 64, 128)), get((3, 3, 128, 256)),
               get((4, 4, 256, 512)).reshape(4096, 512, order='F').T,
               get((1, 1, 512, 10)).reshape(512, 10, order='F').T]
    biases = [get((1, 1, c)).reshape(-1) for c in (64, 128, 256, 512, 10)]
    return weights, biases, get((32, 32, 3))


def matlab_round(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantize(x, wl, fl):
    q = np.clip(matlab_round(x * 2.**fl), -2**(wl-1), 2**(wl-1)-1)
    return q * 2.**(-fl)


def choose_fl(x, wl):
    m = np.max(np.abs(x))
    if m == 0:
        return 0
    lower, upper = -2**(wl-1), 2**(wl-1)-1
    candidate = wl - 1 - int(np.ceil(np.log2(m)))
    while np.any((matlab_round(x * 2.**candidate) < lower) |
                 (matlab_round(x * 2.**candidate) > upper)):
        candidate -= 1
    best_error, best_fl = np.inf, candidate
    while True:
        step = 2.**(-candidate)
        clipping = np.sum(np.maximum(lower*step-x, 0)**2 +
                          np.maximum(x-upper*step, 0)**2)
        if clipping >= best_error:
            return best_fl
        error = np.sum((quantize(x, wl, candidate)-x)**2)
        if error < best_error:
            best_error, best_fl = error, candidate
        candidate += 1


def quantize_tensor(x, wl, channel_axis=None):
    if channel_axis is None:
        return quantize(x, wl, choose_fl(x, wl))
    fl = np.array([choose_fl(np.take(x, c, axis=channel_axis), wl)
                   for c in range(x.shape[channel_axis])])
    shape = [1] * x.ndim
    shape[channel_axis] = len(fl)
    return quantize(x, wl, fl.reshape(shape))


def conv(x, weights, bias):
    h, w, channels = x.shape
    padded = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    patches = np.vstack([sliding_window_view(padded[:, :, c], (3, 3))
                         .transpose(2, 3, 0, 1).reshape(9, h*w, order='F')
                         for c in range(channels)])
    y = weights.reshape(-1, weights.shape[3], order='F').T @ patches + bias[:, None]
    return y.T.reshape(h, w, -1, order='F')


def inference(img, weights, biases, mean, fixed=True):
    x = img - mean
    if fixed:
        x = quantize(x, 32, ACT_FL[0])
    for i in range(3):
        x = np.maximum(conv(x, weights[i], biases[i]), 0)
        if fixed:
            x = quantize(x, 32, ACT_FL[i+1])
        x = np.maximum(np.maximum(x[::2, ::2], x[1::2, ::2]),
                       np.maximum(x[::2, 1::2], x[1::2, 1::2]))
    for i in (3, 4):
        x = weights[i] @ x.reshape(-1, order='F') + biases[i]
        if fixed:
            x = quantize(x, 32, ACT_FL[i+1])
    prob = np.exp(x - x.max())
    prob /= prob.sum()
    assert np.all(np.isfinite(prob)) and np.all(prob >= 0) and abs(prob.sum()-1) < 1e-12
    return prob.argmax()


def main(core_wl=16, output_file='experiment1.csv', configurations=None):
    weights, biases, mean = read_model()
    data = loadmat(ROOT / 'test_image.mat')
    imgs = data['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(float)/255
    labels = data['labels'].reshape(-1)
    baseline = np.array([inference(img, weights, biases, mean, fixed=False) for img in imgs])
    print(f'Floating-point reference: {np.mean(baseline == labels)*100:.2f}%', flush=True)
    total_values = sum(w.size+b.size for w, b in zip(weights, biases)) + mean.size
    rows = []
    if configurations is None:
        configurations = [([wl, core_wl, core_wl, core_wl, wl], wl) for wl in (16, 20, 24)]
    for wls, other_wl in configurations:
        qw = [quantize_tensor(w, wl, 3 if i < 3 else 0)
              for i, (w, wl) in enumerate(zip(weights, wls))]
        qb = [quantize_tensor(b, other_wl) for b in biases]
        qm = quantize_tensor(mean, other_wl)
        predicted = np.array([inference(img, qw, qb, qm) for img in imgs])
        correct = int(np.sum(predicted == labels))
        bits = sum(w.size*wl+b.size*other_wl for w, b, wl in zip(weights, biases, wls))
        bits += mean.size*other_wl
        row = dict(other_wl=other_wl, wl_w=' '.join(map(str, wls)), wl_b=other_wl,
                   wl_m=other_wl, wl_a=32, per_channel=True, correct=correct,
                   num_images=len(labels), accuracy_percent=100*correct/len(labels),
                   changed_predictions=int(np.sum(predicted != baseline)),
                   average_bits=bits/total_values, remaining_percent=100*bits/(32*total_values))
        rows.append(row)
        print(f'WL_W={wls}, other WL={other_wl}: {correct}/{len(labels)}, accuracy={row["accuracy_percent"]:.2f}%, '
              f'changed={row["changed_predictions"]}, average bits={row["average_bits"]:.6f}', flush=True)
    with (RESULTS / output_file).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
