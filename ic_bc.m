function [theta0, y1, y2, y3] = ic_bc(x, t)

    theta0 = sys_ic(x);
    y1 = theta_1(t);
    y2 = theta_2(t);
    y3 = theta_3();

function theta0 = sys_ic(x)
    % b1 = -1.278;
    % b2 = -0.518;
    % b3 = 0.967;
    % b4 = 0.829;
    % theta0 = b1.*x.^3 + b2.*x.^2 + b3.*x + b4;
    theta0 = 0.95238*sin(pi.*x);

function y1 = theta_1(t)
        y1 = 0.0;

function y2 = theta_2(t)
        % y2 = t;
        y2 = 0.0;


function y3 = theta_3()
    y3 = 0.0;

