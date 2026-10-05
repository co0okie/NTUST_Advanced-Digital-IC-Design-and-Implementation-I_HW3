function HW3T(file)
% HW3T  Official test for HW3.
%   HW3T               tests HW3.m in the current folder
%   HW3T(file)         tests another file, e.g. HW3T('D:\M11XXXXXX\HW3.m')
% Only STUDENT_ID and the part after "Your code starts here" are taken from
% the file; the data load, checks and grading are this file's own copy, so
% the result is the one used for grading. HW2_model.mat and test_image.mat
% must be in the same folder as the tested file. Any rule violation only
% prints "Error".

    if nargin < 1
        file = 'HW3.m';
    end
    file = char(file);
    if ~isfile(file)
        fprintf('Error\n');
        return;
    end
    src = fileread(file);

    id = regexp(src, '^\s*STUDENT_ID\s*=\s*[''"](\w+)[''"]', 'tokens', 'once', 'lineanchors');
    k  = strfind(src, '%% ======================= Your code starts here');
    if isempty(id) || numel(k) ~= 1
        fprintf('Error\n');
        return;
    end
    id      = id{1};
    student = src(k:end);

    me = fileread([mfilename('fullpath') '.m']);
    a  = strfind(me, ['%<' 'OFFICIAL>']);
    b  = strfind(me, ['%</' 'OFFICIAL>']);
    official = me(a(end) + numel('%<OFFICIAL>') : b(end) - 1);
    official = regexprep(official, '(?m)^%# ?', '');

    if ~code_is_allowed(student, official, id)
        fprintf('Error\n');
        return;
    end
    official = strrep(official, 'M11XXXXXX', id);

    folder = fileparts(file);
    if isempty(folder)
        folder = pwd;
    end
    runFile = fullfile(folder, 'HW3T_run.m');
    fid = fopen(runFile, 'w', 'n', 'UTF-8');
    fwrite(fid, [official student], 'char');
    fclose(fid);
    oldDir  = cd(folder);
    cleanup = onCleanup(@() finish(oldDir, runFile));
    clear('HW3T_run');
    rehash;
    try
        evalin('base', 'HW3T_run');
    catch err
        if ~strcmp(err.identifier, 'HW3:check')
            rethrow(err);               % a bug in the student's own code
        end
        fprintf('Error\n');
    end
end

function finish(oldDir, runFile)
    cd(oldDir);
    if exist(runFile, 'file')
        delete(runFile);
    end
end

function ok = code_is_allowed(student, official, id)
% Static rules for the student's part of HW3.m.
    code = strip_code(student);

    % 1. Nothing but functions (no script code that runs after grading).
    first = regexp(code, '^\s*function\>', 'once', 'lineanchors');
    if isempty(first) || ~isempty(strtrim(code(1:first-1)))
        ok = false;
        return;
    end

    % 2. The two required functions exist.
    defs = regexp(code, ['^\s*function\s+(?:\[[^\]]*\]\s*=\s*|\w+\s*=\s*)?' ...
                         '(\w+)'], 'tokens', 'lineanchors');
    defs = cellfun(@(c) c{1}, defs, 'UniformOutput', false);
    mine = {id, [id '_quantize']};
    if ~all(ismember(mine, defs))
        ok = false;
        return;
    end

    % 3. No local function may replace anything the grading code uses.
    used = regexp(strip_code(official), '[A-Za-z]\w*', 'match');
    if any(ismember(setdiff(defs, mine), used))
        ok = false;
        return;
    end

    % 4. No NN library, file access, or state kept outside Q (a local
    %    function with the same name is fine).
    banned = { ...
        'predict', 'classify', 'activations', 'forward', 'minibatchpredict', ...
        'trainNetwork', 'trainnet', 'dlnetwork', 'dlarray', 'dlconv', ...
        'fullyconnect', 'maxpool', 'avgpool', 'relu', 'softmax', ...
        'layerGraph', 'assembleNetwork', 'dlquantizer', 'quantize', 'calibrate', ...
        'load', 'save', 'matfile', 'fopen', 'fileread', 'importdata', ...
        'readlines', 'writelines', 'readmatrix', 'writematrix', 'readtable', ...
        'writetable', 'readcell', 'writecell', 'csvread', 'csvwrite', ...
        'dlmread', 'dlmwrite', 'xlsread', 'xlswrite', 'imread', 'imwrite', ...
        'datastore', 'webread', 'websave', ...
        'global', 'persistent', 'assignin', 'evalin', 'eval', 'evalc', ...
        'feval', 'builtin', 'str2func', ...
        'setappdata', 'getappdata', 'setpref', 'getpref', 'addpref', ...
        'setenv', 'getenv', 'set', 'get', 'groot', 'figure', 'gcf', 'gca', ...
        'findobj', 'findall', 'timer', 'timerfind', 'timerfindall', ...
        'memoize', 'clipboard', 'java', 'javaObject', 'javaMethod', 'py', ...
        'system', 'dos', 'unix', 'addpath', 'rmpath'};
    words = regexp(code, '(?<![\w.])[A-Za-z]\w*', 'match');
    ok = isempty(setdiff(intersect(words, banned), defs));
end

function out = strip_code(src)
% Remove comments and the contents of string literals.
    lines  = regexp(src, '\r?\n', 'split');
    inBlk  = false;
    for li = 1:numel(lines)
        L = lines{li};
        t = strtrim(L);
        if inBlk
            inBlk = ~strcmp(t, '%}');
            lines{li} = '';
            continue;
        elseif strcmp(t, '%{')
            inBlk = true;
            lines{li} = '';
            continue;
        end
        res = '';
        i = 1;
        n = numel(L);
        while i <= n
            c = L(i);
            if c == '%' || (i + 2 <= n && strcmp(L(i:i+2), '...'))
                break;
            end
            isStr = c == '"' || (c == '''' && ...
                    (isempty(res) || ~any(res(end) == ['_)]}.''' '0':'9' 'a':'z' 'A':'Z'])));
            if isStr
                j = i + 1;
                while j <= n
                    if L(j) == c
                        if j < n && L(j+1) == c
                            j = j + 2;
                            continue;
                        end
                        break;
                    end
                    j = j + 1;
                end
                res = [res c c];                                    %#ok<AGROW>
                i = j + 1;
            else
                res = [res c];                                      %#ok<AGROW>
                i = i + 1;
            end
        end
        lines{li} = res;
    end
    out = strjoin(lines, newline);
end

% Official copy of HW3.m up to "Your code starts here" (do not change).
%<OFFICIAL>
%# %% HW3 - CIFAR-10 NN fixed-point simulation (based on HW2)
%# % Two functions must be provided, named after STUDENT_ID (e.g. M11XXXXXX):
%# %   Q    = M11XXXXXX_quantize(net)   called ONCE, returns the fixed-point model
%# %   prob = M11XXXXXX(img, Q)         called per image, uses ONLY Q
%# %     img  : 32x32x3 double, pixel value in [0,1]
%# %     prob : 10x1 probability (NOT the category)
%# %
%# % Q must hold every parameter of the given model (input Mean, and Weights /
%# % Bias of every conv and FC layer), each ONLY quantised -- no retraining,
%# % no pruning, no other change to the values:
%# %   * A fixed-point tensor is a struct with fields
%# %        q  : integer array, same size as the original parameter
%# %        WL : word length (bits, scalar)
%# %        FL : fraction length (scalar, or one per output channel)
%# %     q must equal min(max(round(w .* 2.^FL), -2^(WL-1)), 2^(WL-1)-1),
%# %     where w is the original value.  real value = q .* 2.^(-FL)
%# %     cost = numel(q)*WL bits (WL and FL themselves are free)
%# %   * A parameter kept in floating point must equal the original; 32 bits each.
%# %   * Anything else (e.g. activation FL, text) is not counted in the size,
%# %     but is limited to 256 values (numbers + characters) in total.
%# %   * Q must not contain objects or function handles (e.g. the net itself).
%# % Only STUDENT_ID and the part after "Your code starts here" are yours; that
%# % part may contain functions only. HW3T.m grades with its own copy of the
%# % rest, and any rule violation only prints "Error".
%# % Your code must not use NN library functions, read or write files, or keep
%# % data outside Q. These names may not appear in your code (unless you define
%# % a function with that name yourself):
%# %   predict classify activations forward minibatchpredict trainNetwork
%# %   trainnet dlnetwork dlarray dlconv fullyconnect maxpool avgpool relu
%# %   softmax layerGraph assembleNetwork dlquantizer quantize calibrate
%# %   load save matfile fopen fileread importdata readlines writelines
%# %   readmatrix writematrix readtable writetable readcell writecell csvread
%# %   csvwrite dlmread dlmwrite xlsread xlswrite imread imwrite datastore
%# %   webread websave global persistent assignin evalin eval evalc feval
%# %   builtin str2func setappdata getappdata setpref getpref addpref setenv
%# %   getenv set get groot figure gcf gca findobj findall timer timerfind
%# %   timerfindall memoize clipboard java javaObject javaMethod py system dos
%# %   unix addpath rmpath
%# % Your functions may not reuse a name used by the grading code above.
%# clear; clc;
%#
%# STUDENT_ID = 'M11XXXXXX';   % TODO 0: your student ID
%#
%# %% ======================= Data load (do not change) =======================
%# S   = load('HW2_model.mat');
%# net = S.convnet;                          % use net.Layers(k).Weights / Bias ...
%#
%# D      = load('test_image.mat');          % data: 1000x3072 uint8, labels: 1000x1 uint8
%# numImg = size(D.data, 1);
%# labels = double(D.labels);                % class 0 ~ 9
%# labelText = ["airplane","automobile","bird","cat","deer", ...
%#              "dog","frog","horse","ship","truck"];
%#
%# imgs = zeros(32, 32, 3, numImg);
%# for n = 1:numImg
%#     % CIFAR row (R,G,B planes, row-major) -> 32x32x3 image, scaled to [0,1]
%#     imgs(:,:,:,n) = double(permute(reshape(D.data(n,:), 32, 32, 3), [2 1 3])) / 255;
%# end
%# % example: imshow(imgs(:,:,:,1)); title(labelText(labels(1)+1));
%# %% ==========================================================================
%#
%# %% ======================= Grading (do not change) ==========================
%# ACC_MIN    = 80;    % accuracy must be higher than this (%)
%# HALF_BITS  = 16;    % average bits per value must be <= this (size <= 50%)
%# HALF_SCORE = 70;    % score at HALF_BITS
%# FULL_BITS  = 2.5;   % average bits at or below which score = 100
%# % In between, the score grows linearly with log2(compression ratio):
%# % every halving of the model size adds the same number of points.
%# %% ==========================================================================
%#
%# %% ======================= Evaluation (do not change) =======================
%# % Original parameters (FP32 reference: 32 bits per value)
%# ref = struct('name', {'input Mean'}, 'w', {double(net.Layers(1).Mean)});
%# for k = 1:numel(net.Layers)
%#     if isprop(net.Layers(k), 'Weights')
%#         ref(end+1) = struct('name', [net.Layers(k).Name ' Weights'], ...
%#                             'w', double(net.Layers(k).Weights));   %#ok<SAGROW>
%#         ref(end+1) = struct('name', [net.Layers(k).Name ' Bias'], ...
%#                             'w', double(net.Layers(k).Bias));      %#ok<SAGROW>
%#     end
%# end
%# refBits = 32 * sum(arrayfun(@(r) numel(r.w), ref));
%#
%# quantizeFcn = str2func([STUDENT_ID '_quantize']);
%# inferFcn    = str2func(STUDENT_ID);
%#
%# Q = quantizeFcn(net);
%# try
%#     [qBits, qRows, hits, other] = fx_model_size(Q, 'Q', ref);
%# catch
%#     error('HW3:check', 'Error');
%# end
%# if any(hits == 0)
%#     error('HW3:check', 'Error');
%# end
%# if other > 256
%#     error('HW3:check', 'Error');
%# end
%#
%# pred = zeros(numImg, 1);
%# tic;
%# for n = 1:numImg
%#     prob = inferFcn(imgs(:,:,:,n), Q);
%#     if ~isequal(size(prob), [10 1])
%#         error('HW3:check', 'Error');
%#     end
%#     if abs(sum(prob) - 1) > 1e-4 || any(prob < 0)
%#         error('HW3:check', 'Error');
%#     end
%#     [~, k] = max(prob);
%#     pred(n) = k - 1;                      % class 0 ~ 9
%# end
%# t = toc;
%#
%# C = zeros(10);                            % row = true, col = predicted
%# for n = 1:numImg
%#     C(labels(n)+1, pred(n)+1) = C(labels(n)+1, pred(n)+1) + 1;
%# end
%#
%# fprintf('\n #  Class        Images  Accuracy\n');
%# fprintf('--  ----------  ------  --------\n');
%# for k = 1:10
%#     fprintf('%2d  %-10s  %6d  %7.2f%%\n', k-1, labelText(k), ...
%#             sum(C(k,:)), 100*C(k,k)/max(sum(C(k,:)), 1));
%# end
%# fprintf('--  ----------  ------  --------\n');
%# acc = mean(pred == labels) * 100;
%# fprintf('    %-10s  %6d  %7.2f%%\n', 'Total', numImg, acc);
%# fprintf('\nTime : %.2f s\n', t);
%#
%# fprintf('\n%-26s  %-16s  %4s  %12s\n', 'Tensor', 'Size', 'WL', 'Bytes');
%# fprintf('%s\n', repmat('-', 1, 64));
%# for k = 1:size(qRows, 1)
%#     fprintf('%-26s  %-16s  %4s  %12.1f\n', qRows{k,1}, qRows{k,2}, qRows{k,3}, qRows{k,4}/8);
%# end
%# fprintf('%s\n', repmat('-', 1, 64));
%# remain  = 100 * qBits / refBits;
%# avgBits = 32 * qBits / refBits;
%# fprintf('FP32 model size   : %12.1f KB\n', refBits/8/1024);
%# fprintf('Fixed-point size  : %12.1f KB\n', qBits/8/1024);
%# fprintf('Average bits      : %12.2f   bits / value (FP32 = 32)\n', avgBits);
%# fprintf('Compression ratio : %12.2f x\n', 32 / avgBits);
%# fprintf('Remaining size    : %11.2f %%\n', remain);
%# fprintf('Size reduction    : %11.2f %%\n', 100 - remain);
%#
%# if acc <= ACC_MIN
%#     score = 0;
%#     why   = sprintf('accuracy not above %d%%', ACC_MIN);
%# elseif avgBits > HALF_BITS
%#     score = 0;
%#     why   = 'model not compressed to 50% or less';
%# else
%#     score = HALF_SCORE + (100 - HALF_SCORE) * log2(HALF_BITS / avgBits) / log2(HALF_BITS / FULL_BITS);
%#     score = min(100, score);
%#     why   = sprintf('%d at %g bits, 100 at %g bits', HALF_SCORE, HALF_BITS, FULL_BITS);
%# end
%# fprintf('Score             : %12.1f   (%s)\n', score, why);
%#
%# %% ======================= Evaluator helpers (do not change) ================
%#
%# function [bits, rows, hits, other] = fx_model_size(v, path, ref)
%# % Storage cost of everything in v, checking each parameter against ref.
%# %   rows  : {path, size, WL, bits} for every parameter tensor
%# %   hits  : how many times each ref parameter was found
%# %   other : number of values that are not model parameters
%#     bits  = 0;
%#     rows  = cell(0, 4);
%#     hits  = zeros(1, numel(ref));
%#     other = 0;
%#     if isstruct(v) && isfield(v, 'q') && isscalar(v)
%#         r = fx_check(v, path, ref);
%#         hits(r) = 1;
%#         bits = numel(v.q) * v.WL;
%#         rows = {path, mat2str(size(v.q)), num2str(v.WL), bits};
%#         if ~isempty(setdiff(fieldnames(v), {'q', 'WL', 'FL'}))
%#             error('HW3:check', 'Error');
%#         end
%#     elseif isstruct(v) || iscell(v)
%#         f = {};
%#         if isstruct(v), f = fieldnames(v); end
%#         for i = 1:numel(v)
%#             if iscell(v)
%#                 kids = {v{i}};  names = {sprintf('%s{%d}', path, i)};
%#             else
%#                 kids = cellfun(@(n) v(i).(n), f, 'UniformOutput', false);
%#                 idx  = '';
%#                 if numel(v) > 1, idx = sprintf('(%d)', i); end
%#                 names = strcat(path, idx, '.', f);
%#             end
%#             for j = 1:numel(kids)
%#                 [b, r, h, o] = fx_model_size(kids{j}, names{j}, ref);
%#                 bits = bits + b;  rows = [rows; r];                 %#ok<AGROW>
%#                 hits = hits + h;  other = other + o;
%#             end
%#         end
%#     elseif isnumeric(v) || islogical(v)
%#         r = find(arrayfun(@(t) isequal(size(t.w), size(v)), ref));
%#         if ~isempty(r) && numel(v) > 1
%#             if ~isequal(double(v), ref(r).w)
%#                 error('HW3:check', 'Error');
%#             end
%#             bits = 32 * numel(v);
%#             hits(r) = 1;
%#             rows = {path, mat2str(size(v)), 'fp32', bits};
%#         else
%#             other = numel(v);
%#         end
%#     elseif ischar(v) || isstring(v)
%#         other = strlength(string(v));
%#         other = sum(other(:));
%#     else
%#         error('HW3:check', 'Error');
%#     end
%# end
%#
%# function r = fx_check(v, path, ref)
%# % A fixed-point tensor must be an original parameter, rounded to nearest and
%# % saturated -- nothing else. Returns the index of that parameter in ref.
%#     WL = v.WL;  q = double(v.q);  FL = double(v.FL);
%#     if ~isscalar(WL) || WL ~= round(WL) || WL < 1 || WL > 32
%#         error('HW3:check', 'Error');
%#     end
%#     if isempty(FL) || any(FL(:) ~= round(FL(:)))
%#         error('HW3:check', 'Error');
%#     end
%#     r = find(arrayfun(@(t) numel(t.w) == numel(q), ref));
%#     if isempty(r)
%#         error('HW3:check', 'Error');
%#     end
%#     w = ref(r).w;
%#     if ~isequal(size(q), size(w))
%#         error('HW3:check', 'Error');
%#     end
%#     for d = 1:ndims(FL)
%#         if size(FL, d) ~= 1 && size(FL, d) ~= size(w, d)
%#             error('HW3:check', 'Error');
%#         end
%#     end
%#     expect = min(max(round(w .* 2.^FL), -2^(WL-1)), 2^(WL-1) - 1);
%#     bad = find(q ~= expect, 1);
%#     if ~isempty(bad)
%#         error('HW3:check', 'Error');
%#     end
%# end
%</OFFICIAL>
