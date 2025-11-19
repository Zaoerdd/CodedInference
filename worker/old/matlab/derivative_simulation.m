% computation scenario
B = 1;
% input shape
[C_i, H_i, W_i] = deal(64, 224, 224);
% output shape
[C_o, H_o, W_o] = deal(64, 224, 224);

[kernel_size, stride, padding] = deal(3, 1, 1);

conv_params = [B, C_i, H_i, W_i, C_o, H_o, W_o, kernel_size, stride, padding];

[mu_m, theta_m] = deal(5e9, 5e-9);

mu_cmp = 1e9;
theta_cmp = 1e-9;

theta0_tr = 1e-9;
mus_tr = [1e8,2e8,3e8,4e8,5e8,6e8,7e8,8e8,9e8,1e9];

% mu0tr = mustr(0);
mus_results_n = [];
mus_results_k = [];
partial_ns = [];
partial_ks = [];

for i = 1:length(mus_tr)
    mu0_tr = mus_tr(i);
    % system_params = [mum, thetam, mu0tr, theta0tr, mucmp, thetacmp, mu0tr, theta0tr];
    [mu0_rec, mu0_sen] = deal(mu0_tr);
    [theta0_rec, theta0_sen] = deal(theta0_tr);

    % [h1,h2,h3,h4,h5,h6,h7,h8] = problem_params(system_params, conv_params);
    Iov = prod([C_i, H_i, kernel_size - stride]);
    IW = prod([C_i, H_i, W_o, stride]);
    O = prod([C_o, H_o, W_o]);
    Nc = prod([2.0, C_o, H_o, C_i, kernel_size, kernel_size, W_o]); % 这个用int32表示会溢出了
    
    h1 = prod([2.0,Iov,1/mu_m+theta_m]);
    h2 = prod([2.0,IW,1/mu_m+theta_m]) + prod([4,Iov,theta0_rec]);
    h3 = prod([2.0,O,1/mu_m+theta_m]);
    h4 = prod([4,IW,theta0_rec]) + prod([4,O,theta0_sen]);
    h5 = prod([Nc,theta_cmp]);
    h6 = prod([4,Iov,1/mu0_rec]);
    h7 = prod([4,IW,1/mu0_rec]) + prod([4,O,1/mu0_sen]);
    h8 = prod([Nc,1/mu_cmp]);

    % [n_optimal, k_optimal, min_value] = optimize_biconvex(h1,h2,h3,h4,h5,h6,h7,h8);
    % 定义目标函数 L(n, k)
    L = @(x) h1*x(1)*x(2) + h2*x(1) + h3*x(2) + h4*x(1)/x(2) + h5/x(2) + ...
        h6*x(1)*log(x(1)/(x(1)-x(2))) + h7*x(1)/x(2)*log(x(1)/(x(1)-x(2))) + ...
        h8/x(2)*log(x(1)/(x(1)-x(2)));

    % 定义不等式约束
    % nonlcon = @(x) [x(1) - x(2); x(2) - 1]; % n > k, k > 1
    nonlcon = @(x) myNonlinearConstraints(x);

    % 定义变量下界和上界
    lb = [3; 2]; % 变量下界
    ub = [Inf; Inf]; % 变量上界

    % 调用 fmincon 求解
    x0 = [15; 8]; % 初始点
    options = optimoptions('fmincon', 'Display', 'off'); % 设置选项
    [x, min_value] = fmincon(L, x0, [], [], [], [], lb, ub, nonlcon, options);

    % 输出结果
    n_optimal = x(1);
    k_optimal = x(2);
    mus_results_n = [mus_results_n, n_optimal];
    mus_results_k = [mus_results_k, k_optimal];
    partial_n = cal_partial_n(h1,h2,h3,h4,h5,h6,h7,h8,n_optimal,k_optimal);
    partial_k = cal_partial_k(h1,h2,h3,h4,h5,h6,h7,h8,n_optimal,k_optimal);
    partial_ns = [partial_ns, partial_n];
    partial_ks = [partial_ks, partial_k];
    fprintf('%f %f %f %f %f\n', min_value, n_optimal, partial_n, k_optimal, partial_k)
end

% 根据计算参数计算关于n的一阶偏导
function [partial_n] = cal_partial_n(h1,h2,h3,h4,h5,h6,h7,h8,n,k)
    partial_n = h1*k + h2 + h4/k + (h6 + h7/k)*(log(n/(n-k)) - k/(n-k)) - h8/(n*(n-k));
end

% 根据计算参数计算关于k的一阶偏导
function [partial_k] = cal_partial_k(h1,h2,h3,h4,h5,h6,h7,h8,n,k)
    partial_k = n*h1 + h3 - (n*h4 + h5)/k^2 + n*h6/(n-k) + (n*h7 + h8)*(-1/k^2 * log(n/(n-k)) + 1/k*(n-k));
end
    
    
% 自定义非线性约束函数
function [c, ceq] = myNonlinearConstraints(x)
    % 将不等式转换为形式化约束
    c = [x(1) - x(2);  % n > k
         x(2) - 2];    % k > 1
    ceq = []; % 没有等式约束
end

