%% HW3 - CIFAR-10 NN fixed-point simulation (based on HW2)
% Two functions must be provided, named after STUDENT_ID (e.g. M11XXXXXX):
%   Q    = M11XXXXXX_quantize(net)   called ONCE, returns the fixed-point model
%   prob = M11XXXXXX(img, Q)         called per image, uses ONLY Q
%     img  : 32x32x3 double, pixel value in [0,1]
%     prob : 10x1 probability (NOT the category)
%
% Q must hold every parameter of the given model (input Mean, and Weights /
% Bias of every conv and FC layer), each ONLY quantised -- no retraining,
% no pruning, no other change to the values:
%   * A fixed-point tensor is a struct with fields
%        q  : integer array, same size as the original parameter
%        WL : word length (bits, scalar)
%        FL : fraction length (scalar, or one per output channel)
%     q must equal min(max(round(w .* 2.^FL), -2^(WL-1)), 2^(WL-1)-1),
%     where w is the original value.  real value = q .* 2.^(-FL)
%     cost = numel(q)*WL bits (WL and FL themselves are free)
%   * A parameter kept in floating point must equal the original; 32 bits each.
%   * Anything else (e.g. activation FL, text) is not counted in the size,
%     but is limited to 256 values (numbers + characters) in total.
%   * Q must not contain objects or function handles (e.g. the net itself).
% Only STUDENT_ID and the part after "Your code starts here" are yours; that
% part may contain functions only. HW3T.m grades with its own copy of the
% rest, and any rule violation only prints "Error".
% Your code must not use NN library functions, read or write files, or keep
% data outside Q. These names may not appear in your code (unless you define
% a function with that name yourself):
%   predict classify activations forward minibatchpredict trainNetwork
%   trainnet dlnetwork dlarray dlconv fullyconnect maxpool avgpool relu
%   softmax layerGraph assembleNetwork dlquantizer quantize calibrate
%   load save matfile fopen fileread importdata readlines writelines
%   readmatrix writematrix readtable writetable readcell writecell csvread
%   csvwrite dlmread dlmwrite xlsread xlswrite imread imwrite datastore
%   webread websave global persistent assignin evalin eval evalc feval
%   builtin str2func setappdata getappdata setpref getpref addpref setenv
%   getenv set get groot figure gcf gca findobj findall timer timerfind
%   timerfindall memoize clipboard java javaObject javaMethod py system dos
%   unix addpath rmpath
% Your functions may not reuse a name used by the grading code above.
clear; clc;

STUDENT_ID = 'M11507401';   % TODO 0: your student ID

%% ======================= Data load (do not change) =======================
S   = load('HW2_model.mat');
net = S.convnet;                          % use net.Layers(k).Weights / Bias ...

D      = load('test_image.mat');          % data: 1000x3072 uint8, labels: 1000x1 uint8
numImg = size(D.data, 1);
labels = double(D.labels);                % class 0 ~ 9
labelText = ["airplane","automobile","bird","cat","deer", ...
             "dog","frog","horse","ship","truck"];

imgs = zeros(32, 32, 3, numImg);
for n = 1:numImg
    % CIFAR row (R,G,B planes, row-major) -> 32x32x3 image, scaled to [0,1]
    imgs(:,:,:,n) = double(permute(reshape(D.data(n,:), 32, 32, 3), [2 1 3])) / 255;
end
% example: imshow(imgs(:,:,:,1)); title(labelText(labels(1)+1));
%% ==========================================================================

%% ======================= Grading (do not change) ==========================
ACC_MIN    = 80;    % accuracy must be higher than this (%)
HALF_BITS  = 16;    % average bits per value must be <= this (size <= 50%)
HALF_SCORE = 70;    % score at HALF_BITS
FULL_BITS  = 2.5;   % average bits at or below which score = 100
% In between, the score grows linearly with log2(compression ratio):
% every halving of the model size adds the same number of points.
%% ==========================================================================

%% ======================= Evaluation (do not change) =======================
% Original parameters (FP32 reference: 32 bits per value)
ref = struct('name', {'input Mean'}, 'w', {double(net.Layers(1).Mean)});
for k = 1:numel(net.Layers)
    if isprop(net.Layers(k), 'Weights')
        ref(end+1) = struct('name', [net.Layers(k).Name ' Weights'], ...
                            'w', double(net.Layers(k).Weights));   %#ok<SAGROW>
        ref(end+1) = struct('name', [net.Layers(k).Name ' Bias'], ...
                            'w', double(net.Layers(k).Bias));      %#ok<SAGROW>
    end
end
refBits = 32 * sum(arrayfun(@(r) numel(r.w), ref));

quantizeFcn = str2func([STUDENT_ID '_quantize']);
inferFcn    = str2func(STUDENT_ID);

Q = quantizeFcn(net);
try
    [qBits, qRows, hits, other] = fx_model_size(Q, 'Q', ref);
catch
    error('HW3:check', 'Error');
end
if any(hits == 0)
    error('HW3:check', 'Error');
end
if other > 256
    error('HW3:check', 'Error');
end

pred = zeros(numImg, 1);
tic;
for n = 1:numImg
    prob = inferFcn(imgs(:,:,:,n), Q);
    if ~isequal(size(prob), [10 1])
        error('HW3:check', 'Error');
    end
    if abs(sum(prob) - 1) > 1e-4 || any(prob < 0)
        error('HW3:check', 'Error');
    end
    [~, k] = max(prob);
    pred(n) = k - 1;                      % class 0 ~ 9
end
t = toc;

C = zeros(10);                            % row = true, col = predicted
for n = 1:numImg
    C(labels(n)+1, pred(n)+1) = C(labels(n)+1, pred(n)+1) + 1;
end

fprintf('\n #  Class        Images  Accuracy\n');
fprintf('--  ----------  ------  --------\n');
for k = 1:10
    fprintf('%2d  %-10s  %6d  %7.2f%%\n', k-1, labelText(k), ...
            sum(C(k,:)), 100*C(k,k)/max(sum(C(k,:)), 1));
end
fprintf('--  ----------  ------  --------\n');
acc = mean(pred == labels) * 100;
fprintf('    %-10s  %6d  %7.2f%%\n', 'Total', numImg, acc);
fprintf('\nTime : %.2f s\n', t);

fprintf('\n%-26s  %-16s  %4s  %12s\n', 'Tensor', 'Size', 'WL', 'Bytes');
fprintf('%s\n', repmat('-', 1, 64));
for k = 1:size(qRows, 1)
    fprintf('%-26s  %-16s  %4s  %12.1f\n', qRows{k,1}, qRows{k,2}, qRows{k,3}, qRows{k,4}/8);
end
fprintf('%s\n', repmat('-', 1, 64));
remain  = 100 * qBits / refBits;
avgBits = 32 * qBits / refBits;
fprintf('FP32 model size   : %12.1f KB\n', refBits/8/1024);
fprintf('Fixed-point size  : %12.1f KB\n', qBits/8/1024);
fprintf('Average bits      : %12.2f   bits / value (FP32 = 32)\n', avgBits);
fprintf('Compression ratio : %12.2f x\n', 32 / avgBits);
fprintf('Remaining size    : %11.2f %%\n', remain);
fprintf('Size reduction    : %11.2f %%\n', 100 - remain);

if acc <= ACC_MIN
    score = 0;
    why   = sprintf('accuracy not above %d%%', ACC_MIN);
elseif avgBits > HALF_BITS
    score = 0;
    why   = 'model not compressed to 50% or less';
else
    score = HALF_SCORE + (100 - HALF_SCORE) * log2(HALF_BITS / avgBits) / log2(HALF_BITS / FULL_BITS);
    score = min(100, score);
    why   = sprintf('%d at %g bits, 100 at %g bits', HALF_SCORE, HALF_BITS, FULL_BITS);
end
fprintf('Score             : %12.1f   (%s)\n', score, why);

%% ======================= Evaluator helpers (do not change) ================

function [bits, rows, hits, other] = fx_model_size(v, path, ref)
% Storage cost of everything in v, checking each parameter against ref.
%   rows  : {path, size, WL, bits} for every parameter tensor
%   hits  : how many times each ref parameter was found
%   other : number of values that are not model parameters
    bits  = 0;
    rows  = cell(0, 4);
    hits  = zeros(1, numel(ref));
    other = 0;
    if isstruct(v) && isfield(v, 'q') && isscalar(v)
        r = fx_check(v, path, ref);
        hits(r) = 1;
        bits = numel(v.q) * v.WL;
        rows = {path, mat2str(size(v.q)), num2str(v.WL), bits};
        if ~isempty(setdiff(fieldnames(v), {'q', 'WL', 'FL'}))
            error('HW3:check', 'Error');
        end
    elseif isstruct(v) || iscell(v)
        f = {};
        if isstruct(v), f = fieldnames(v); end
        for i = 1:numel(v)
            if iscell(v)
                kids = {v{i}};  names = {sprintf('%s{%d}', path, i)};
            else
                kids = cellfun(@(n) v(i).(n), f, 'UniformOutput', false);
                idx  = '';
                if numel(v) > 1, idx = sprintf('(%d)', i); end
                names = strcat(path, idx, '.', f);
            end
            for j = 1:numel(kids)
                [b, r, h, o] = fx_model_size(kids{j}, names{j}, ref);
                bits = bits + b;  rows = [rows; r];                 %#ok<AGROW>
                hits = hits + h;  other = other + o;
            end
        end
    elseif isnumeric(v) || islogical(v)
        r = find(arrayfun(@(t) isequal(size(t.w), size(v)), ref));
        if ~isempty(r) && numel(v) > 1
            if ~isequal(double(v), ref(r).w)
                error('HW3:check', 'Error');
            end
            bits = 32 * numel(v);
            hits(r) = 1;
            rows = {path, mat2str(size(v)), 'fp32', bits};
        else
            other = numel(v);
        end
    elseif ischar(v) || isstring(v)
        other = strlength(string(v));
        other = sum(other(:));
    else
        error('HW3:check', 'Error');
    end
end

function r = fx_check(v, path, ref)
% A fixed-point tensor must be an original parameter, rounded to nearest and
% saturated -- nothing else. Returns the index of that parameter in ref.
    WL = v.WL;  q = double(v.q);  FL = double(v.FL);
    if ~isscalar(WL) || WL ~= round(WL) || WL < 1 || WL > 32
        error('HW3:check', 'Error');
    end
    if isempty(FL) || any(FL(:) ~= round(FL(:)))
        error('HW3:check', 'Error');
    end
    r = find(arrayfun(@(t) numel(t.w) == numel(q), ref));
    if isempty(r)
        error('HW3:check', 'Error');
    end
    w = ref(r).w;
    if ~isequal(size(q), size(w))
        error('HW3:check', 'Error');
    end
    for d = 1:ndims(FL)
        if size(FL, d) ~= 1 && size(FL, d) ~= size(w, d)
            error('HW3:check', 'Error');
        end
    end
    expect = min(max(round(w .* 2.^FL), -2^(WL-1)), 2^(WL-1) - 1);
    bad = find(q ~= expect, 1);
    if ~isempty(bad)
        error('HW3:check', 'Error');
    end
end

%% ======================= Your code starts here ============================
% 1. Set STUDENT_ID at the top, and rename the two functions M11XXXXXX_quantize
%    and M11XXXXXX below to the same ID.
% 2. Fill in TODO 1 ~ 4. With the default 16-bit settings the accuracy should
%    stay about the same as HW2, and the score should be 70.
% 3. Lower the word lengths in TODO 5 (and improve TODO 1 if you like) to make
%    the model smaller while keeping the accuracy above 80%.
% Run HW3T to see the official result before you submit.

function Q = M11507401_quantize(net)
% Convert the HW2 network into a fixed-point model. Called once.

    % TODO 5 -----------------------------------------------------------------
    % Word lengths in bits. Smaller = smaller model = higher score.
    WL_W = [16 3 3 2 16];       % weights of conv1 conv2 conv3 fc1 fc2
    WL_B = 16;                  % biases
    WL_M = 16;                  % input mean image
    WL_A = 32;                  % activations (not counted in the model size)
    PER_CHANNEL = true;         % true: one FL per output channel for weights
    % ------------------------------------------------------------------------

    % Max |activation| of the float model on the test images, measured after:
    % input - mean, relu1, relu2, relu3, fc1, fc2.
    ACT_MAX = [0.8617 1.217 2.855 2.65 2.749 43.01];

    Q.actWL = WL_A;
    ops = struct('type', {}, 'W', {}, 'B', {}, 'mean', {}, 'aFL', {});
    k   = 0;                    % conv / FC layer index
    for i = 1:numel(net.Layers)
        L    = net.Layers(i);
        name = strsplit(class(L), '.');
        op   = struct('type', '', 'W', [], 'B', [], 'mean', [], 'aFL', []);
        switch name{end}
            case 'ImageInputLayer'
                op.type = 'input';
                op.mean = fx_tensor(L.Mean, WL_M, 0);
            case 'Convolution2DLayer'           % 3x3, stride 1, padding 'same'
                k = k + 1;
                op.type = 'conv';
                op.W = fx_tensor(L.Weights, WL_W(k), 4 * PER_CHANNEL);
                op.B = fx_tensor(L.Bias, WL_B, 0);
            case 'ReLULayer'
                op.type = 'relu';
            case 'MaxPooling2DLayer'            % 2x2, stride 2
                op.type = 'pool';
            case 'FullyConnectedLayer'
                k = k + 1;
                op.type = 'fc';
                op.W = fx_tensor(L.Weights, WL_W(k), 1 * PER_CHANNEL);
                op.B = fx_tensor(L.Bias, WL_B, 0);
            case 'SoftmaxLayer'
                op.type = 'softmax';
            case {'DropoutLayer', 'ClassificationOutputLayer'}
                continue;                       % nothing to do at inference
            otherwise
                error('Unhandled layer type: %s', name{end});
        end
        ops(end+1) = op;                                            %#ok<AGROW>
    end

    % Activations are quantised after the input, every ReLU, and every FC
    % that is not followed by a ReLU.
    a = 0;
    for j = 1:numel(ops)
        fcOut = strcmp(ops(j).type, 'fc') && ~strcmp(ops(j+1).type, 'relu');
        if any(strcmp(ops(j).type, {'input', 'relu'})) || fcOut
            a = a + 1;
            ops(j).aFL = WL_A - 1 - ceil(log2(ACT_MAX(a)));
        end
    end
    Q.ops = ops;
end

function prob = M11507401(img, Q)
% Fixed-point simulation of the network. Use ONLY Q.
    x = img;
    for j = 1:numel(Q.ops)
        op = Q.ops(j);
        switch op.type
            case 'input'
                x = x - fx_value(op.mean);
            case 'conv'
                x = conv3x3_same(x, fx_value(op.W), fx_value(op.B));
            case 'relu'
                x = max(x, 0);
            case 'pool'
                x = maxpool2x2(x);
            case 'fc'
                x = fx_value(op.W) * x(:) + fx_value(op.B);
            case 'softmax'
                e = exp(x - max(x(:)));
                x = e / sum(e(:));
        end
        if ~isempty(op.aFL)
            % TODO 4 -------------------------------------------------------------
            % Quantise the activation x to Q.actWL bits with fraction length
            % op.aFL (use fx_int and fx_real).
            x = fx_real(fx_int(x, Q.actWL, op.aFL), op.aFL);
            % --------------------------------------------------------------------
        end
    end
    prob = double(x(:));
end

%% ---------------------------------------------------------------- helpers

function FL = choose_fl(x, WL)
% Fraction length used to store the values x with WL bits.
    % TODO 1 -----------------------------------------------------------------
    % Basic: the largest FL for which max(abs(x)) does not overflow.
    % Better (optional): try a few larger FL as well and keep the one with the
    % smallest error sum((fx_real(fx_int(x, WL, FL), FL) - x).^2).
    x = double(x(:));
    m = max(abs(x));
    if m == 0
        FL = 0;
        return;
    end

    qmin = -2^(WL-1);
    qmax = 2^(WL-1)-1;
    candidate = WL - 1 - ceil(log2(m));
    while any(round(x .* 2.^candidate) < qmin | ...
              round(x .* 2.^candidate) > qmax)
        candidate = candidate - 1;
    end

    bestError = Inf;
    FL = candidate;
    while true
        step = 2.^(-candidate);
        clipError = sum(max(qmin*step - x, 0).^2 + ...
                        max(x - qmax*step, 0).^2);
        if clipError >= bestError
            break;
        end
        q = min(max(round(x .* 2.^candidate), qmin), qmax);
        totalError = sum((q .* step - x).^2);
        if totalError < bestError
            bestError = totalError;
            FL = candidate;
        end
        candidate = candidate + 1;
    end
    % ------------------------------------------------------------------------
end

function q = fx_int(x, WL, FL)
% Signed WL-bit integer of x with fraction length FL.
    % TODO 2 -----------------------------------------------------------------
    % Scale x by 2^FL, round to the NEAREST integer (use round), then saturate
    % to [-2^(WL-1), 2^(WL-1)-1]. FL may be a scalar or an array that
    % broadcasts against x (one FL per channel).
    q = min(max(round(double(x) .* 2.^FL), -2^(WL-1)), 2^(WL-1)-1);
    % ------------------------------------------------------------------------
end

function v = fx_real(q, FL)
% Real value of integer q with fraction length FL.
    % TODO 3 -----------------------------------------------------------------
    v = double(q) .* 2.^(-FL);
    % ------------------------------------------------------------------------
end

function t = fx_tensor(x, WL, chDim)
% Fixed-point tensor {q, WL, FL}. chDim = 0: one FL for the whole tensor;
% otherwise one FL per slice along dimension chDim (output channel).
    x = double(x);
    if chDim == 0
        FL = choose_fl(x(:), WL);
    else
        n   = size(x, chDim);
        sz  = ones(1, max(2, ndims(x)));  sz(chDim) = n;
        FL  = zeros(sz);
        idx = repmat({':'}, 1, ndims(x));
        for c = 1:n
            idx{chDim} = c;
            s = x(idx{:});
            FL(c) = choose_fl(s(:), WL);
        end
    end
    t.q  = fx_int(x, WL, FL);
    t.WL = WL;
    t.FL = FL;
end

function v = fx_value(t)
% Real value of a fixed-point tensor (or a parameter kept in floating point).
    if isnumeric(t)
        v = t;
    else
        v = fx_real(t.q, t.FL);
    end
end

function Y = conv3x3_same(X, W, B)
% 3x3 convolution (cross-correlation), stride 1, zero padding 1.
%   X : H x W x Cin,  W : 3 x 3 x Cin x Cout,  B : 1 x 1 x Cout
    [H, Wd, Cin] = size(X);
    [kh, kw, ~, Cout] = size(W);
    Hp = H + 2;  Wp = Wd + 2;
    Xp = zeros(Hp, Wp, Cin);
    Xp(2:H+1, 2:Wd+1, :) = X;

    inPatch = (1:kh)' + ((1:kw) - 1) * Hp;              % kh x kw
    inPatch = inPatch(:) + (0:Cin-1) * (Hp * Wp);       % (kh*kw) x Cin
    origin  = (0:H-1)' + (0:Wd-1) * Hp;                 % H x W
    cols    = Xp(inPatch(:) + origin(:)');              % (kh*kw*Cin) x (H*W)

    Y = reshape(W, kh * kw * Cin, Cout).' * cols + B(:);
    Y = reshape(Y.', H, Wd, Cout);
end

function Y = maxpool2x2(X)
    Y = max(max(X(1:2:end, 1:2:end, :), X(2:2:end, 1:2:end, :)), ...
            max(X(1:2:end, 2:2:end, :), X(2:2:end, 2:2:end, :)));
end
