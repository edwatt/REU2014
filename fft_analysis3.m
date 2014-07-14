#read fft file in and separate into frequency and power
#then analyze the data and plot
src =  csvread('fft_samples/fft_20140707-114835.csv');
x = src(1,:);
y = src(2,:);
der  = y(2:end) - y(1:end-1);
der = [0 der];
#figure;
#plot(x,y);
#line = x * 0 + mean(y);
avg_power = mean(y);
y_t = zeros(columns(y)) + avg_power;

thres = 7;

for i = 1:columns(y)
	if (abs(der(i)) <= thres)
		y_t(i) = y(i);
	endif
endfor

figure;
plot(x,y_t);



