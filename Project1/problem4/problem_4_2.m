% Problem 4_2 2018016244 추현욱

clear; % clear workspace
% Generate random c, A, b
c = rand(100, 1); % 10x1 vector
A = randn(55, 100) * 5 - 1.2; % 8x10 matrix
b = -1 + (1 + 1) * rand(55, 1); % 8x1 vector

% Set parameters of linprog(f, A, b, Aeq, Beq, lb, ub)
f = c;      % set f
Aeq = [];   % set Aeq. There's no Aeq
Beq = [];   % set Beq. There's no Beq
lb = zeros(size(f, 1), size(f, 2)); % x_i >= 0 (lower boubnd)
ub = [];    % There's no upper bound

% Use linprog() function
[x, fval] = linprog(f, A, b, Aeq, Beq, lb, ub); % get optimal solution, optimal cost


disp('Optimal Solution')
disp(x)
disp('Optimal cost')
disp(fval)