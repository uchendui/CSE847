d = importdata("ad_data.mat");
feat = importdata("feature_name.mat");

pars = [0.01; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
num_features = []
aucs = []
for n =1:size(pars,1)
    [w,c] = logistic_l1_train(d.X_train, d.y_train, pars(n));
    
    % Evaluate on the test data set
    predictions = sigmoid( d.X_test * w + c); 
    
    [X,Y,T,AUC] = perfcurve(d.y_test, predictions, 1);
    
    % Transform predictions
    %predictions(predictions < 0.5) = 0;
    %predictions(predictions >= 0.5) = 1;
    
    d.y_test(d.y_test==-1) = 0;
    aucs  = [aucs AUC]
    num_features = [num_features nnz(w)]
end

figure()
plot(pars,aucs, '-r.')
title("Sparse Logistic Regression AUC")
xlabel("Regularization Parameter Value")
ylabel("AUC")

figure()
plot(pars,num_features, '-r.')
title("Sparse Logistic Regression Number of Features Selected")
xlabel("Regularization Parameter Value")
ylabel("Number of Features")