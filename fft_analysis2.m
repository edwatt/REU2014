#read fft file in and separate into frequency and power
#then analyze the data and plot
src =  csvread('fft_samples/fft_20140710-155000.csv');
x = src(1,:);
y = src(2,:);
der  = y(2:end) - y(1:end-1);
der = [0 der];
#figure;
#plot(x,y);
#line = x * 0 + mean(y);
avg_power = mean(y);
y_t = zeros(columns(y));
der_thres = 6;
der_t = zeros(columns(y)) + avg_power;
for i = 1:columns(y)
	if(y(i) < avg_power)
		y_t(i) = avg_power;
	else
		y_t(i) = y(i);
		#if(abs(der(i)) <= der_thres)
		#	der_t(i) = y(i);
		#end
	end
end

plot(x,y_t,x,der_t);
#plot(x,y,'-',x,line,'-');
#figure;
#plot(x,der);

#y_t = zeros(columns(y));

#thres = 5;

#for i = 1:columns(y)
#	if (der(i) >= thres)
#		y_t(i) = 1;
#	elseif (der(i) <= -thres)
#		y_t(i) = -1;
#	endif
#endfor

#figure;
#plot(x,y_t);



