function [h1,h2,h3,h4,h5,h6,h7,h8] = problem_params(system_params,conv_params)
%ESTIMATED_LATENCY 此处显示有关此函数的摘要
%   此处显示详细说明
    [B, C_i, H_i, W_i, C_o, H_o, W_o, kernel_size, stride, padding] = deal(conv_params);
    [mu_m, theta_m, mu0_rec, theta0_rec, mu_cmp, theta_cmp, mu0_sen, theta0_sen] = deal(system_params);
    
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
end

