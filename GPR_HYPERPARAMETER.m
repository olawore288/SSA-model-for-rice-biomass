%% ==================== Clear workspace ====================
clear; clc;

%% ==================== Read training and testing data ====================
train_data = readtable('train_data.xlsx');
test_data  = readtable('test_data.xlsx');

% Separate features (X) and target (y)
X_train = train_data{:,1:end-1};
y_train = train_data{:,end};

X_test  = test_data{:,1:end-1};
y_test  = test_data{:,end};

%% ==================== Grid Search (manual, respecting lower bounds) ====================
lengthScaleRange = logspace(-1,1,5);  % length scale
sigmaRange       = linspace(0.96,2,5); % Sigma above MATLAB lower bound

bestR2_grid = -Inf;
gpr_grid_best = [];

for l = lengthScaleRange
    for s = sigmaRange
        try
            gpr = fitrgp(X_train, y_train, ...
                'KernelFunction','squaredexponential', ...
                'Standardize',true, ...
                'KernelParameters',[l,1], ...   % signal std = 1 (can adjust)
                'Sigma', s);

            y_pred_train = predict(gpr, X_train);
            R2_val = 1 - sum((y_train - y_pred_train).^2)/sum((y_train - mean(y_train)).^2);

            if R2_val > bestR2_grid
                bestR2_grid = R2_val;
                gpr_grid_best = gpr;
            end
        catch
            continue
        end
    end
end
fprintf('Grid Search Best R2 = %.4f\n', bestR2_grid);

%% ==================== Random Search ====================
numRandomTrials = 20;
bestRMSE_rand = Inf;
gpr_rand_best = [];

% Set random seed for reproducibility
rng(1);

for i = 1:numRandomTrials

    % Randomly sample hyperparameters
    l = 0.96 + (10 - 0.96)*rand();    % Kernel scale
    s = 0.96 + (2 - 0.96)*rand();     % Noise sigma

    try
        % Train GPR model
        gpr = fitrgp(X_train, y_train, ...
            'KernelFunction','squaredexponential', ...
            'Standardize',true, ...
            'KernelParameters',[l,1], ...
            'Sigma', s);

        % Five-fold cross-validation
        cvModel = crossval(gpr,'KFold',5);

        % Cross-validation RMSE
        rmseCV = sqrt(kfoldLoss(cvModel,'LossFun','mse'));

        % Keep best model
        if rmseCV < bestRMSE_rand
            bestRMSE_rand = rmseCV;
            gpr_rand_best = gpr;

            bestParamsRand = struct( ...
                'KernelScale', l, ...
                'SignalStd', 1, ...
                'NoiseSigma', s);
        end

    catch
        continue
    end

end

fprintf('\nBest Random Search Parameters:\n');
fprintf('Kernel Scale = %.4f\n',bestParamsRand.KernelScale);
fprintf('Signal Std   = %.4f\n',bestParamsRand.SignalStd);
fprintf('Noise Sigma  = %.4f\n',bestParamsRand.NoiseSigma);
fprintf('5-Fold CV RMSE = %.4f\n',bestRMSE_rand);

%% ==================== Predictions ====================
y_train_pred_grid = predict(gpr_grid_best, X_train);
y_test_pred_grid  = predict(gpr_grid_best, X_test);

y_train_pred_rand = predict(gpr_rand_best, X_train);
y_test_pred_rand  = predict(gpr_rand_best, X_test);

%% ==================== Evaluation metrics ====================
RMSE = @(y,yhat) sqrt(mean((y - yhat).^2));
RRMSE = @(y,yhat) RMSE(y,yhat)/mean(y);
RSE = @(y,yhat) sum((y - yhat).^2);
RRSE = @(y,yhat) sqrt(sum((y - yhat).^2)/sum((y - mean(y)).^2));
RAE = @(y,yhat) sum(abs(y - yhat))/sum(abs(y - mean(y)));
R2fun  = @(y,yhat) 1 - sum((y - yhat).^2)/sum((y - mean(y)).^2);
PI  = @(y,yhat) R2fun(y,yhat)/RRMSE(y,yhat);

computeMetrics = @(y,yhat) struct( ...
    'R2', R2fun(y,yhat), ...
    'RMSE', RMSE(y,yhat), ...
    'RRMSE', RRMSE(y,yhat), ...
    'RSE', RSE(y,yhat), ...
    'RRSE', RRSE(y,yhat), ...
    'RAE', RAE(y,yhat), ...
    'PI', PI(y,yhat));

metrics_grid_train = computeMetrics(y_train, y_train_pred_grid);
metrics_grid_test  = computeMetrics(y_test, y_test_pred_grid);

metrics_rand_train = computeMetrics(y_train, y_train_pred_rand);
metrics_rand_test  = computeMetrics(y_test, y_test_pred_rand);

%% ==================== Save predictions ====================
train_data.Grid_Y_Pred = y_train_pred_grid;
train_data.Rand_Y_Pred = y_train_pred_rand;
test_data.Grid_Y_Pred  = y_test_pred_grid;
test_data.Rand_Y_Pred  = y_test_pred_rand;

writetable(train_data,'train_predictions.xlsx');
writetable(test_data,'test_predictions.xlsx');

%% ==================== Display metrics ====================
disp('--- Grid Search Metrics ---');
disp('Training Set:'); disp(metrics_grid_train);
disp('Testing Set:'); disp(metrics_grid_test);

disp('--- Random Search Metrics ---');
disp('Training Set:'); disp(metrics_rand_train);
disp('Testing Set:'); disp(metrics_rand_test);

%% ==================== Display optimized hyperparameters ====================
disp('--- Grid Search Optimized Hyperparameters ---');
disp(['Kernel Scale: ', num2str(gpr_grid_best.KernelInformation.KernelParameters(1))]);
disp(['Signal Std:    ', num2str(gpr_grid_best.KernelInformation.KernelParameters(2))]);
disp(['Noise Sigma:   ', num2str(gpr_grid_best.Sigma)]);

disp('--- Random Search Optimized Hyperparameters ---');
disp(['Kernel Scale: ', num2str(gpr_rand_best.KernelInformation.KernelParameters(1))]);
disp(['Signal Std:    ', num2str(gpr_rand_best.KernelInformation.KernelParameters(2))]);
disp(['Noise Sigma:   ', num2str(gpr_rand_best.Sigma)]);

%% ==================== Plot predictions vs actual ====================
figure;
subplot(2,2,1);
scatter(y_train, y_train_pred_grid,'b','filled'); hold on;
plot([min(y_train) max(y_train)], [min(y_train) max(y_train)], 'r--', 'LineWidth',1.5);
xlabel('Actual SSA'); ylabel('Predicted SSA'); title('Grid Search - Train'); grid on;

subplot(2,2,2);
scatter(y_test, y_test_pred_grid,'b','filled'); hold on;
plot([min(y_test) max(y_test)], [min(y_test) max(y_test)], 'r--', 'LineWidth',1.5);
xlabel('Actual SSA'); ylabel('Predicted SSA'); title('Grid Search - Test'); grid on;

subplot(2,2,3);
scatter(y_train, y_train_pred_rand,'g','filled'); hold on;
plot([min(y_train) max(y_train)], [min(y_train) max(y_train)], 'r--', 'LineWidth',1.5);
xlabel('Actual SSA'); ylabel('Predicted SSA'); title('Random Search - Train'); grid on;

subplot(2,2,4);
scatter(y_test, y_test_pred_rand,'g','filled'); hold on;
plot([min(y_test) max(y_test)], [min(y_test) max(y_test)], 'r--', 'LineWidth',1.5);
xlabel('Actual SSA'); ylabel('Predicted SSA'); title('Random Search - Test'); grid on;

sgtitle('Predicted vs Actual SSA');

disp('Prediction complete. Metrics, hyperparameters, and plots are ready.');
