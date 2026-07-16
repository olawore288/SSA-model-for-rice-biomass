%% Clear workspace
clear; clc;

%% ================================
% Load Training and Testing Data
trainData = readtable('train_data.xlsx');
testData  = readtable('test_data.xlsx');

% Separate inputs and target
X_train = trainData{:, 1:end-1};
y_train = trainData{:, end};

X_test  = testData{:, 1:end-1};
y_test  = testData{:, end};

%% ================================
% Define hyperparameter grid for GBRT
hyperGrid.NumLearningCycles = [50 100 200];
hyperGrid.LearnRate = [0.01 0.05 0.1 0.2];
hyperGrid.MaxNumSplits = [2 5 10 20];

%% ==========================================================
% 1️⃣ GRID SEARCH OPTIMIZATION
fprintf('\nRunning Grid Search Optimization...\n');

bestRMSE = Inf;

for nTrees = hyperGrid.NumLearningCycles
    for lr = hyperGrid.LearnRate
        for maxSplit = hyperGrid.MaxNumSplits
            
            t = templateTree('MaxNumSplits', maxSplit);
            model = fitrensemble(X_train, y_train, ...
                'Method','LSBoost', ...
                'NumLearningCycles', nTrees, ...
                'LearnRate', lr, ...
                'Learners', t);
            
            cvmodel = crossval(model,'KFold',5);
            rmse_cv = sqrt(kfoldLoss(cvmodel,'LossFun','mse'));
            
            if rmse_cv < bestRMSE
                bestRMSE = rmse_cv;
                bestModelGrid = model;
                bestParamsGrid = struct('NumTrees',nTrees,...
                                        'LearnRate',lr,...
                                        'MaxNumSplits',maxSplit);
            end
        end
    end
end

fprintf('Best Grid Params: Trees=%d | LR=%.3f | Splits=%d\n', ...
    bestParamsGrid.NumTrees, ...
    bestParamsGrid.LearnRate, ...
    bestParamsGrid.MaxNumSplits);

%% ==========================================================
% 2️⃣ RANDOM SEARCH OPTIMIZATION
fprintf('\nRunning Random Search Optimization...\n');

numRandomTrials = 20;
bestRMSERandom = Inf;

rng(1); % Reproducibility

for i = 1:numRandomTrials
    
    nTrees   = randsample(hyperGrid.NumLearningCycles,1);
    lr       = hyperGrid.LearnRate(randi(length(hyperGrid.LearnRate)));
    maxSplit = hyperGrid.MaxNumSplits(randi(length(hyperGrid.MaxNumSplits)));
    
    t = templateTree('MaxNumSplits', maxSplit);
    model = fitrensemble(X_train, y_train, ...
        'Method','LSBoost', ...
        'NumLearningCycles', nTrees, ...
        'LearnRate', lr, ...
        'Learners', t);
    
    cvmodel = crossval(model,'KFold',5);
    rmse_cv = sqrt(kfoldLoss(cvmodel,'LossFun','mse'));
    
    if rmse_cv < bestRMSERandom
        bestRMSERandom = rmse_cv;
        bestModelRandom = model;
        bestParamsRandom = struct('NumTrees',nTrees,...
                                  'LearnRate',lr,...
                                  'MaxNumSplits',maxSplit);
    end
end

fprintf('Best Random Params: Trees=%d | LR=%.3f | Splits=%d\n', ...
    bestParamsRandom.NumTrees, ...
    bestParamsRandom.LearnRate, ...
    bestParamsRandom.MaxNumSplits);

%% ==========================================================
% 3️⃣ PREDICTIONS FOR BOTH MODELS

% -------- GRID MODEL ----------
yPredTrain_Grid = predict(bestModelGrid, X_train);
yPredTest_Grid  = predict(bestModelGrid, X_test);

% -------- RANDOM MODEL ----------
yPredTrain_Rand = predict(bestModelRandom, X_train);
yPredTest_Rand  = predict(bestModelRandom, X_test);

%% ==========================================================
% 4️⃣ EVALUATION METRICS FUNCTION

computeMetrics = @(yTrue,yPred) struct( ...
    'R2', 1 - sum((yTrue - yPred).^2)/sum((yTrue - mean(yTrue)).^2), ...
    'RMSE', sqrt(mean((yTrue - yPred).^2)), ...
    'RRMSE', sqrt(mean((yTrue - yPred).^2)) / mean(yTrue), ...
    'RSE', sum((yTrue - yPred).^2)/sum((yTrue - mean(yTrue)).^2), ...
    'RRSE', sqrt(sum((yTrue - yPred).^2)/sum((yTrue - mean(yTrue)).^2)), ...
    'RAE', sum(abs(yTrue - yPred))/sum(abs(yTrue - mean(yTrue))), ...
    'PI', (1 - sqrt(mean((yTrue - yPred).^2)) / mean(yTrue)) ...
    );

%% ==========================================================
% 5️⃣ COMPUTE PERFORMANCE

% ---- GRID ----
metricsTrain_Grid = computeMetrics(y_train, yPredTrain_Grid);
metricsTest_Grid  = computeMetrics(y_test, yPredTest_Grid);

% ---- RANDOM ----
metricsTrain_Rand = computeMetrics(y_train, yPredTrain_Rand);
metricsTest_Rand  = computeMetrics(y_test, yPredTest_Rand);

%% ==========================================================
% 6️⃣ DISPLAY RESULTS

fprintf('\n========== GRID SEARCH PERFORMANCE ==========\n');
disp('Training Performance:'); disp(metricsTrain_Grid);
disp('Testing Performance:');  disp(metricsTest_Grid);

fprintf('\n========== RANDOM SEARCH PERFORMANCE ==========\n');
disp('Training Performance:'); disp(metricsTrain_Rand);
disp('Testing Performance:');  disp(metricsTest_Rand);

%% ==========================================================
% 7️⃣ SAVE RESULTS

trainData.SSA_Pred_Grid  = yPredTrain_Grid;
trainData.SSA_Pred_Rand  = yPredTrain_Rand;

testData.SSA_Pred_Grid   = yPredTest_Grid;
testData.SSA_Pred_Rand   = yPredTest_Rand;

writetable(trainData,'SSA_GBRT_Train_Results.xlsx');
writetable(testData,'SSA_GBRT_Test_Results.xlsx');

disp('All predictions and performance evaluations saved successfully.');


%% ==========================================================
% 8️⃣ FEATURE IMPORTANCE (ROBUST VERSION)

impGrid = predictorImportance(bestModelGrid);
impRandom = predictorImportance(bestModelRandom);

% Normalize
impGrid = impGrid ./ sum(impGrid);
impRandom = impRandom ./ sum(impRandom);

% Get predictor names directly from model
featureNames = bestModelGrid.PredictorNames';

fprintf('\n========== FEATURE IMPORTANCE (GRID) ==========\n');
T_grid = table(featureNames, impGrid(:), ...
    'VariableNames', {'Feature','NormalizedImportance'});
disp(T_grid);

fprintf('\n========== FEATURE IMPORTANCE (RANDOM) ==========\n');
featureNames_rand = bestModelRandom.PredictorNames';

T_rand = table(featureNames_rand, impRandom(:), ...
    'VariableNames', {'Feature','NormalizedImportance'});
disp(T_rand);

% Plot Grid Importance
figure;
bar(impGrid);
set(gca,'XTickLabel',featureNames);
xtickangle(45);
ylabel('Normalized Importance');
title('Feature Importance - GBRT (Grid Search)');
grid on;
