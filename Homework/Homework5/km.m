function clusters = km(R, K)
changed = true
%Select K points as the initial centroids
centroids = R(randperm(size(R,1),K), :);
figure()
while changed == true
    clusters = cell(1,K);
    % Form K clusters by assigning all points to the closest centroid.
    for i=1:size(R,1)
        point = R(i,:);
        distances = pdist2(point,centroids);
        [~, mem] = min(distances);
        clusters{mem} = [clusters{mem}; point];
    end    
    
    % Recompute the centroid of each cluster
    new_centroids = zeros(K,2);
    for i=1:K
        new_centroids(i, :) = mean(clusters{i});
    end
    
    if  isequal(new_centroids, centroids)
        changed=false;
    end
    centroids = new_centroids;
end




end
