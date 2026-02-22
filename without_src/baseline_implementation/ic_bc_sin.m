function [theta0, y1, theta0_0, y3] = ic_bc_sin(x, t)

    theta0 = sys_ic(x);
    y1 = theta_1(t);
    theta0_0 = theta_2(t);
    y3 = theta_3(t);
end
function theta0 = sys_ic(x)
    theta0 = 0.95238*sin(pi.*x);
end
function y1 = theta_1(t)
        y1 = 0.0;
end
function theta0_0 = theta_2(t)
        theta0_0 = 0.0;

end
function y3 = theta_3(t)
    y3 = 0.0;
end
