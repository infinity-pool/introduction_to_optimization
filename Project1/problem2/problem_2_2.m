% Problem 2_2 2018016244 추현욱
A = load('./datas/A.mat').A; % Load predefined A
Q = load('./datas/Q.mat').Q; % Load predefined Q
b = load('./datas/b.mat').b; % Load predefined b
x0 = ones(1000, 1); % Set initial condition [1, 1, ..., 1]
f = @(x) 0.5 * x' * Q * x - b' * x; % Define cost function

% Define options for fmincon
options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'iter', 'MaxIterations', 100, 'MaxFunctionEvaluations', 2e5);

% No linear inequality or equality constraints
A_ineq = [];
b_ineq = [];
A_eq = [];
b_eq = [];

% No nonlinear constraints
nonl_constraints = [];

% No upper, lower bounds
upper_bound = [];
lower_bound = [];

% Run fmincon
[x_optimal, fval, exitflag, output] = fmincon(f, x0, A_ineq, b_ineq, A_eq, b_eq, lower_bound, upper_bound, nonl_constraints, options);

% Display the results
disp(['Iterations: ', num2str(output.iterations)]);
disp(['Optimal cost: ', num2str(fval)]);

save('./datas/x_optimal_by_2.mat', 'x_optimal'); % Save x_optimal vector

% Result
%{
Iterations: 32
Optimal cost: -0.099632
%}