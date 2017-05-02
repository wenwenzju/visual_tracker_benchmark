clear,clc;
%color map
pn = 100;
% map = [ones(pn/2,1), [0:1/(pn/2-1):1]', zeros(pn/2,1)];
% map = [map;[[1-1/(pn/2-1):-1/(pn/2-1):0]',ones(pn/2-1,1),zeros(pn/2-1,1)]];
map = [ones(pn/2,1), [0:1/(pn/2):1-1/(pn/2)]', zeros(pn/2,1)];
map = [map;[[1-1/(pn/2):-1/(pn/2):0]',ones(pn/2,1),zeros(pn/2,1)]];
map = flipud(map);

% hs = importdata('hog_scores.txt');
hs = importdata('sdalf_scores.txt');

f1 = figure;
f2 = figure;
% f3 = figure;
imgs_num = 500;
img_path = 'D:\data_seq\Human7\';
k = 1;
hist_num = 30;
aspect_ratio = 0.43;

for i = 1:imgs_num
    s = hs(pn*(i-1)+1:pn*i,1);
    x = hs(pn*(i-1)+1:pn*i,2);
    y = hs(pn*(i-1)+1:pn*i,3);
    z = hs(pn*(i-1)+1:pn*i,4);
%     [idx,C] = kmeans([x,y,s], k);
    [~,ind] = sort(s);
    ind = flip(ind);

    img_name = sprintf('%04d.jpg', i+1);
    img = imread([img_path img_name]);
    
    figure(f1);
    imshow(img),
    hold on
    for j = 1:pn
        if s(ind(j)) < -realmax/10
            continue;
        end
        plot(x(ind(j)), y(ind(j)), '.', 'MarkerSize', 25, 'Color', map(j,:));
    end

    ss = (s-min(s))/(max(s)-min(s));
%     ss = ss/sum(ss);
%     td = randsample(1:length(x), length(x), true, exp(ss));
    
    edges = min([0 cumsum(exp(2*s)/sum(exp(2*s)))'],1); % protect against accumulated round-off
    edges(end) = 1;                 % get the upper edge exact
    u1 = rand/length(x);
    % this works like the inverse of the empirical distribution and returns
    % the interval where the sample is to be found
    [~, td] = histc(u1:1/length(x):1, edges);
    
    tmp = [x,y];
    s_rs = s(td);
    td = tmp(td,:);
    
    clf(f2,'reset');
    figure(f2);
    subplot(2,2,1);
    hx = histogram(x,hist_num,'Normalization','probability');title('x');
    subplot(2,2,2);
    hy = histogram(y,hist_num,'Normalization','probability');title('y');
    
    subplot(2,2,3);
    rs_hx = histogram(td(:,1),hist_num,'Normalization','probability');
    hold on;
%     exps = exp(s);
    exps = ss;
    exps = exps/(sum(exps));
    [~,indx] = sort(x);
%     plot(x(indx),exps(indx),'-.');

    he1 = hist_entropy(rs_hx.Values, 3, 1);
    plot(rs_hx.BinEdges(he1.idx), he1.Values, '-.');

%     dx = bhattacharyya(rs_hx.Values, hx.Values);
%     dx = emd_hat_gd_metric_mex(rs_hx.Values', hx.Values', D, -1);
    title(sprintf('sigmax=%f', var(td(:,1))));
    
    subplot(2,2,4);
    rs_hy = histogram(td(:,2),hist_num,'Normalization','probability');
    hold on;
    [~,indy] = sort(y);
%     plot(y(indy),exps(indy),'-.');

    he2 = hist_entropy(rs_hy.Values, 3,1);
    plot(rs_hy.BinEdges(he2.idx), he2.Values, '-.');

%     dy = bhattacharyya(rs_hy.Values, hy.Values);
%     dy = emd_hat_gd_metric_mex(rs_hy.Values', hy.Values', D, -1);
    title(sprintf('sigmay=%f', var(td(:,2))));
    
%     figure(f3);
%     hist2 = histogram2(x, y, hist_num);
%     hist2_rs = histogram2(td(:,1), td(:,2), hist_num);
%     subplot(1,2,1);
%     bar3(hist2.x, hist2.Values);
%     r1 = gaussian_fit2(x, y, s);
%     title(sprintf('residual=%f', r1));
%     subplot(1,2,2);
%     bar3(hist2_rs.x, hist2_rs.Values);
%     r2 = gaussian_fit2(td(:,1), td(:,2), s_rs);
%     title(sprintf('residual=%f', r2));
end