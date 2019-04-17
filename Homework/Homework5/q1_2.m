clear

% Generate data points for the clusters
mus = [0 5; 4 3];
sigma = [1 0; 0 1];
R = [];
for i=1:2
    R = [R; mvnrnd(mus(i, :),sigma,50)];
end
figure()
plot(R(:,1),R(:,2),'+')
title("Original Data Points")

% K Means clustering
K = 2
colors = ['r'; 'b';'g';'y'];
clusters = km(R, K);
for i=1:K
    plot(clusters{i}(:,1),clusters{i}(:,2),[colors(i) 'o'])
    hold on
end
hold off
title("K Means Clustering")

    
% Implement spectral relaxation for K means
XTX = R * R';
size(XTX);
d = eigs(XTX, K);

% Find corresponding eigenvectors
Y = []
for i=1:size(d,1)
z = null(XTX - d(i).* eye(size(R,1),size(R,1)));
Y = [Y z]
end

clusters = km(Y, K);
plot(Y(:,1),Y(:,2),'o')
title("Spectral transformation")

figure()
for i=1:2
    plot(clusters{i}(:,1),clusters{i}(:,2),[colors(i) 'o'])
    hold on
end
hold off
title("Spectral K Means Clustering")

   
   
   
