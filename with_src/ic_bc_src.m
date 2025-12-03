function [theta0, y1, theta0_0, y3] = ic_bc_src(x, t)

    theta0 = sys_ic(x);
    y1 = theta_1(t);
    theta0_0 = theta_2(t);
    y3 = theta_3(t);
end
function theta0 = sys_ic(x)
    b1 = -1.278;
    b2 = -0.518;
    b3 = 0.967;
    b4 = 0.829;
    theta0 = b1.*x.^3 + b2.*x.^2 + b3.*x + b4;
    %theta0 = 0.95238*sin(pi.*x);
end
function y1 = theta_1(t)
        y1 = 0.0;
end
function theta0_0 = theta_2(t)
        % theta0_0 = t;
        b4 = 0.829;
        theta0_0 = b4;
        %theta0_0 = 0.0;

end
function y3 = theta_3(t)
    y3 = 0.65*(1-exp(-t/0.5));
    
end
