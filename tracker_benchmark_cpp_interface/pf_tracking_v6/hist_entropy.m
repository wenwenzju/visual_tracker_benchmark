function he = hist_entropy(hist, epoch_len, epoch_jump)
hist_len = length(hist);
epoch = 1;
values = zeros(1,ceil((hist_len-epoch_len)/epoch_jump)+1);
idx = zeros(size(values));
hei = 1;
while 1
    if epoch > hist_len
        break;
    end
    lower = epoch;
    upper = epoch + epoch_len - 1;
    if upper > hist_len
        upper = hist_len;
    end
    values(hei) = -sum(hist(lower:upper).*log1(hist(lower:upper)));
    idx(hei) = epoch + epoch_len/2;
    epoch = epoch + epoch_jump;
    hei = hei + 1;
end
he.Values = values/sum(values);
he.idx = floor(idx);

function y = log1(x)
x = x+(x == 0);
y = log(x);