clear,clc;
hog = importdata('hog_scores.txt');
sdalf = importdata('match_scores.txt');
%%
pn = 100;
imgs_num = 500;
f = figure;
w = 0.8;
for i = 1:imgs_num
    clf(f,'reset');
    x = hog(pn*(i-1)+1:pn*i,2);
    y = hog(pn*(i-1)+1:pn*i,3);
    hog_s = hog(pn*(i-1)+1:pn*i, 1);
%     hog_s = (hog_s-min(hog_s))/(max(hog_s)-min(hog_s));
%     hog_s = hog_s./sum(hog_s);
    hog_s = exp(2*hog_s);
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

%%
pn = 100;
imgs_num = 251;
f = figure;
hog_mean = zeros(1,imgs_num);
hog_var = zeros(1, imgs_num);
sdalf_mean = zeros(1,imgs_num);
sdalf_var = zeros(1,imgs_num);
for i = 1:imgs_num
    clf(f,'reset');
    hog_s = hog(pn*(i-1)+1:pn*i, 1);
%     hog_s = exp(2*hog_s);
    hog_mean(i) = mean(hog_s);
    hog_var(i) = var(hog_s);
%     hog_s = hog_s./sum(hog_s);
    
    sdalf_s = sdalf(pn*(i-1)+1:pn*i, 1);
%     sdalf_s = exp(0.6./sdalf_s);
    sdalf_mean(i) = mean(sdalf_s);
    sdalf_var(i) = var(sdalf_s);
%     sdalf_s = sdalf_s./sum(sdalf_s);
end
hold on
% plot(1:imgs_num, hog_mean, '-*', 'LineWidth',2, 'Color', 'r');
% plot(1:imgs_num, hog_mean+hog_var, '-*', 'LineWidth',1, 'Color', 'r');
% plot(1:imgs_num, hog_mean-hog_var, '-*', 'LineWidth',1, 'Color', 'r');

plot(1:imgs_num, sdalf_mean, '-o', 'LineWidth',2, 'Color', 'b');
plot(1:imgs_num, sdalf_mean+sdalf_var, '-o', 'LineWidth',1, 'Color', 'b');
plot(1:imgs_num, sdalf_mean-sdalf_var, '-o', 'LineWidth',1, 'Color', 'b');
xlabel('image i');ylabel('original scores');
title('David3');
legend('hog\_score','hog\_score+var','hog\_score-var','match\_score','match\_score+var','match\_score-var');