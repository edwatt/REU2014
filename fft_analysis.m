#read fft file in and separate into frequency and power
#then analyze the data and plot
src =  csvread('fft_samples/fft_20140710-155000.csv');
x = src(1,:);
y = src(2,:);
avg_power = mean(y);
y_t = zeros(1,floor(columns(y)/7));
x_t = zeros(1,floor(columns(y)/7));
y = y(1:floor(columns(y)/7)*7);
x = x(1:floor(columns(y)/7)*7);


for i = 1:columns(y_t)
	x_t(i) = x((i-1)*7 + 4);
	for j = 1:7
		if (y((i-1)*7 + j) > avg_power)
			y_t(i) = 1;
			break;
		endif
	endfor
endfor

length = 0;
length_thres = 10;

for i = 1:columns(y_t)
	if(y_t(i) == 1)
		length++;
	else
		if(length < length_thres)
			y_t(i-length:i-1) = zeros(1,length);
		endif
		length = 0;
	endif
endfor

		
figure;
plot(x_t,y_t);

