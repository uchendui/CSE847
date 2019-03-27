data = importdata('data.txt');
labels = importdata('labels.txt');

% Convert labels to +1/-1
labels(labels==0) = -1;
data = [data ones(size(data,1),1)];

test_x = data(2001:end,:);
test_y = labels(2001:end,:);
accuracies = [];
batch_sizes = [200; 500; 800; 1000; 1500; 2000];
for n =1:size(batch_sizes,1)
    train_x = data(1:batch_sizes(n),:);
    train_y = labels(1:batch_sizes(n),:);
    
    % Train the logistic regression classifier
    model = logistic_train(train_x, train_y);
    size(train_x);
    
    % Evaluate on the test data set
    predictions = sigmoid( test_x * model); 
    
    % Transform predictions
    predictions(predictions < 0.5) = 0;
    predictions(predictions >= 0.5) = 1;
    
    test_y(test_y==-1) = 0;
    cp = classperf(test_y, predictions);
    accuracies = [accuracies 1-cp.ErrorRate]
end

figure()
plot(batch_sizes, accuracies, '-bo')
title("Logistic Regression Accuracy")
xlabel("Training Set Size")
ylabel("Accuracy")
