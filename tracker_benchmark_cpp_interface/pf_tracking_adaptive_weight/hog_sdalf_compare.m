hog = importdata('hog_scores.txt');
sdalf = importdata('sdalf_scores.txt');

pn = 100;
imgs_num = 500;
f = figure;
w = 0.8;
for i = 1:imgs_num
    clf(f,'reset');
    x = hog(pn*(i-1)+1:pn*i,2);
    y = hog(pn*(i-1)+1:pn*i,3);
    hog_s = hog(pn*(i-1)+1:pn*i, 1);
    hog_s = (hog_s-min(hog_s))/(max(hog_s)-min(hog_s));
%     hog_s = hog_s./sum(hog_s);
    hog_s = exp(3*hog_s);
    hog_s = hog_s./sum(hog_s);
%     m1 = max(hog_s);
    hog_mean_x = sum(hog_s.*x);
    hog_mean_y = sum(hog_s.*y);
    
    sdalf_s = sdalf(pn*(i-1)+1:pn*i, 1);
%     ds = min(sdalf_s);
%     alpha = 2^((log(ds)-m1*sum(log(sdalf_s)))/(m1*pn-1));
%     sdalf_s = sdalf_s./sum(sdalf_s);
%     sdalf_s = exp(m1*m2./sdalf_s);
%     sdalf_s = -log(alpha.*sdalf_s);
    sdalf_s = exp(0.6./sdalf_s);
    sdalf_s = sdalf_s./sum(sdalf_s);
    sdalf_mean_x = sum(sdalf_s.*x);
    sdalf_mean_y = sum(sdalf_s.*y);
    
    weight_mean_x = w*sdalf_mean_x+(1-w)*hog_mean_x;
    weight_mean_y = w*sdalf_mean_y+(1-w)*hog_mean_y;
    
    mean_x = mean(x);
    mean_y = mean(y);
    
    [~,idx] = sort(x);
    subplot(2,1,1),plot(x(idx), hog_s(idx), 'b*');hold on;
    plot(x(idx), sdalf_s(idx), 'g^');
    plot(x(idx), w*sdalf_s(idx)+(1-w)*hog_s(idx), 'ro');
    line([hog_mean_x hog_mean_x], [0 max([hog_s; sdalf_s])], 'Color', 'b');
    line([sdalf_mean_x sdalf_mean_x], [0 max([hog_s; sdalf_s])], 'Color', 'g');
    line([weight_mean_x weight_mean_x], [0 max([hog_s; sdalf_s])], 'Color', 'r');
    line([weight_mean_x weight_mean_x], [0 max([hog_s; sdalf_s])], 'Color', 'r');
    line([mean_x mean_x], [0 max([hog_s; sdalf_s])], 'Color', 'k');
    
    [~,idx] = sort(y);
    subplot(2,1,2), plot(y(idx),hog_s(idx), 'b*'); hold on;
    plot(y(idx), sdalf_s(idx), 'g^');
    plot(y(idx), w*sdalf_s(idx)+(1-w)*hog_s(idx), 'ro');
    line([hog_mean_y hog_mean_y], [0 max([hog_s; sdalf_s])], 'Color', 'b');
    line([sdalf_mean_y sdalf_mean_y], [0 max([hog_s; sdalf_s])], 'Color', 'g');
    line([weight_mean_y weight_mean_y], [0 max([hog_s; sdalf_s])], 'Color', 'r');
    line([mean_y mean_y], [0 max([hog_s; sdalf_s])], 'Color', 'k');
    figure(f);
end