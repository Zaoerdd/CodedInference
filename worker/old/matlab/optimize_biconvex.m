function [n_optimal, k_optimal, min_value] = optimize_biconvex(h1, h2, h3, h4, h5, h6, h7, h8)
    % 定义目标函数 L(n, k)
    L = @(x) h1*x(1)*x(2) + h2*x(1) + h3*x(2) + h4*x(1)/x(2) + h5/x(2) + ...
        h6*x(1)*log(x(1)/(x(1)-x(2))) + h7*x(1)/x(2)*log(x(1)/(x(1)-x(2))) + ...
        h8/x(2)*log(x(1)/(x(1)-x(2)));

    % 定义不等式约束
    nonlcon = @(x) [x(1) - x(2); x(2) - 1]; % n > k, k > 1

    % 定义变量下界和上界
    lb = [1; 1]; % 变量下界
    ub = [Inf; Inf]; % 变量上界

    % 调用 fmincon 求解
    x0 = [3; 2]; % 初始点
    options = optimoptions('fmincon', 'Display', 'iter'); % 设置选项
    [x, min_value] = fmincon(L, x0, [], [], [], [], lb, ub, nonlcon, options);

    % 输出结果
    n_optimal = x(1);
    k_optimal = x(2);
end



