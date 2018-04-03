clear all
close all

file1 = '/home/almul0/Universidad/ityst/17-18/30357 - Laboratorio de señal y comunicaciones/TP61/tiger-costume/sources/javi_uno_16K.wav';
file2 = '/home/almul0/Universidad/ityst/17-18/30357 - Laboratorio de señal y comunicaciones/TP61/tiger-costume/sources/alberto_uno_16K.wav';


[x_javi,fs] = audioread(file1);
[x_alberto,fs] = audioread(file2);

N= 512;
W = hamming(N);


x_javi = x_javi(:,1);
x_alberto = x_alberto(:,1);

s = x_javi;


plot((1:length(s))/fs, s)

L1 = length(s);
T1 = 300*1e-3;
T2 = 10*1e-3; 
N1 = floor(T1*fs);%muestras por bloque
D1 = floor(T2*fs);
indice  = 1:N1-(N1-D1):L1;

noise_power = sum(abs(s(1:floor(1*fs/2)).^2));
display(noise_power)
noise_th1 = 100*noise_power;
noise_th2 = 100*noise_power;
display(noise_th1 )
display(noise_th2)
sflag = 0;
clip  = zeros(size(s));
wp = zeros(size(s));
chunks = [];
for i = 2:(length(indice)-1)    
   fin = min(indice(i)+N1, length(s));
   window_power = sum(abs(s(indice(i):fin).^2));
   if (window_power > noise_th1 && sflag==0)
       sflag = 1;
       clip(indice(i)) = window_power;
       chunks = [ chunks indice(i)];
   elseif (window_power < noise_th2 && sflag==1 )
        clip(indice(i)+N1) = window_power;
        sflag = 0;
        chunks = [ chunks indice(i)+N1 ];
   end
   wp(indice(i)) = window_power;
end
close all
figure
plot((1:length(s))/fs, s)
hold on
yyaxis right
plot((1:length(s))/fs, clip)
%plot((1:length(s))/fs, wp+eps)
legend(sprintf('th1=%.2f',noise_th1),sprintf('th2=%.2f',noise_th2))

display(max(wp))
display(length(chunks))
%%

max_chunk_dur = max(chunks(2:2:end)-chunks(1:2:end));

train_samples = zeros(length(chunks)/2,max_chunk_dur);

ti = 1;
for i = 1:2:length(chunks)
   chunk_dur = chunks(i+1)-chunks(i);
   chunk_diff = max_chunk_dur - chunk_dur;
   if (mod(chunk_diff,2) == 0) 
        off_s = chunk_diff/2;
        off_e = off_s;
   else 
       off_s = floor(chunk_diff/2);
       off_e = off_s+1;
   end
   train_samples(ti,:) = s(chunks(i)-off_s+1:chunks(i+1)+off_e);
   %soundsc(s(chunks(i):chunks(i+1)),fs)
   %pause
   ti = ti+1;
end
%%
figure
for i = 1:size(train_samples,1)
   plot(train_samples(i,:))
   ylim([-1 1])
   soundsc(train_samples(i,:),fs)
   pause
end
%%
x1 = train_samples(5,:);

frame_period = 5;
frame_samples = round(fs*frame_period/1000);

N = 6*frame_samples;
M = 2*frame_samples;

x_w = windower(x,M,N);

x_power = sum(abs(x_w).^2)
th = 0.1;
speech_th = (x_power>th);
speech_index = speech_th*M;
noise_th = x_power<=th;
noise_index = noise_th*M;
% #speech_length = np.transpose(np.concatenate((x_th*M, np.ones(x_th.shape)*M), axis=0))
% #speech_index = tuple(map(int,speech_index))
x_fil = [];
for i=1:length(speech_index)
    x_fil = [x_fil x(peech_index(i):(speech_index(i)+M))]
end
