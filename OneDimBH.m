function [sol] = OneDimBH()

  m = 0;
  x = linspace(0,1,101);
  t = linspace(0,1,101);

  % Run PDE solver with nested functions
  sol = pdepe(m, @OneDimBHpde, @OneDimBHic, @OneDimBHbc, x, t);

  u1 = sol(:,:,1); % solution of system

  % Write solution to file
  fileID = fopen('/Users/guglielmocappellini/Desktop/research/code/pinns-wave/wave-gnn/bioheat_deeponet/gt_bioheat1D.csv', 'w');
  for i = 1:101
      for j = 1:101
          fprintf(fileID,'%6.2f %6.2f %12.8f\n', x(j), t(i), u1(i,j));
      end
  end
  fclose(fileID);

% Plot the solution u1 vs x and t
figure;
surf(x, t, u1, 'EdgeColor', 'none');
xlabel('x');
ylabel('t');
zlabel('u1(x,t)');
title('Solution of 1D Bioheat Equation');
colorbar;
view(2); % Top view for better visualization
  % --------------------------------------------------------------------------
  function [c, f, s] = OneDimBHpde(x, t, u, dudx)
      wb = 0.0005;  % Uses config from outer scope

      a1 = 18.992;
      a2 = 34185.667;
      a3 = 0.0;
      a4 = 3.570;

      c = a1;
      f = dudx;
      s = -wb * a2 * u + a3 * exp(-a4 * x);
  end

  % --------------------------------------------------------------------------
  function u0 = OneDimBHic(x)
      [theta0, ~, ~, ~] = ic_bc(x, 0);
      u0 = theta0;
  end

  % --------------------------------------------------------------------------
  function [pl, ql, pr, qr] = OneDimBHbc(xl, ul, xr, ur, t)
      [~, y1, ~, ~] = ic_bc(xr, t);
      [~, ~, ~, y3] = ic_bc(xl, t);

      a5 = 1.167;
      pl = ul - 0; % Dirichlet boundary condition: u(x=0, t) = 0
      ql = 0;

    %   pl = a5 * (y3 - ul); Robin bc
    %   ql = 1;

      pr = ur - y1;
      qr = 0;
  end
end