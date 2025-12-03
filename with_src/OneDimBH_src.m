function [sol] = OneDimBH_src()

  m = 0;
  x = linspace(0,1,101);
  t = linspace(0,1,101);

  % Run PDE solver with nested functions
  sol = pdepe(m, @OneDimBHpde, @OneDimBHic, @OneDimBHbc, x, t);

  u1 = sol(:,:,1); % solution of system

  % Write solution to file
  fileID = fopen("\Users\Mattia\Dropbox\Mattia\Tesi Magistrale/gt_bioheat1D_src.csv", 'w');
  for i = 1:101
      for j = 1:101
          fprintf(fileID,'%6.2f %6.2f %12.8f\n', x(j), t(i), u1(i,j));
      end
  end
  fclose(fileID);

% Plot the solution u1 vs x and t
figure();
surf(x, t, u1, 'EdgeColor', 'none');
xlabel('x');
ylabel('t');
zlabel('u1(x,t)');
beta = 1;
P =100;
title(sprintf('Solution of 1D Bioheat Equation (beta = %.2f), (Power = %.2f)', beta, P));
colorbar;
saveas(fig, 'bioheat_1D_src.png');


    % --------------------------------------------------------------------------
    function [c, f, s] = OneDimBHpde(x, t, u, dudx)
            wb = 0.0005;  % Uses config from outer scope

            a1 = 18.992;
            a2 = 34185.667;
            a4 = 3.570;

            %a3 computing
            beta = 1;
            P = 100;

            ro = 1000; k = 0.6; Tmin = 21.5; x0 = 0.004; PD = 0.0136;L0 = 0.07;
            v = log(2/(PD-10^(-2)* x0));
            b4 = 0.829;
            y2_0 = 30.2;
            deltaT = (y2_0 - Tmin)/b4;
            
            a3 = ((ro*L0^2)/(k*deltaT))*beta*exp(v*x0);

            c = a1;
            f = dudx;
            s = -wb * a2 * u + P*a3 * exp(-a4 * x);
    end

    % --------------------------------------------------------------------------
    function u0 = OneDimBHic(x)
            % b1 = -1.278;
            % b2 = -0.518;
            % b3 = 0.967;
            % b4 = 0.829;
            % u0 = b1.*x.^3 + b2.*x.^2 + b3.*x + b4;
            u0 = 0;
    end

    % --------------------------------------------------------------------------
    function [pl, ql, pr, qr] = OneDimBHbc(xl, ul, xr, ur, t)
            [~, y1, ~, ~] = ic_bc_src(xr, t);
            [~, ~, ~, y3] = ic_bc_src(xl, t);

            %a5 = 1.167;
            a5 = 175; % Heat transfer coefficient
            pl = -a5 * (-y3 + ul); % Robin bc
            ql = 1;

            pr = ur - y1;        % Dirichlet boundary condition: u(x=1, t) = y1
            qr = 0;
    end
end