clear all
close all
clc

x_file = 'j21.wav';



[x,fs] = audioread(x_file);

N= 512;
W = hamming(N);

x = x(:,1);

% figure
% plot((1:length(x))/fs, x)


T1 = 50*1e-3;
T2 = 10*1e-3; 
N1 = floor(T1*fs);%muestras por bloque
D1 = floor(T2*fs);



L1 = length(x);
indice  = 1:N1-(N1-D1):L1;
noise_power = min([max([sum(abs(x(1:floor(1*fs/2)).^2)), 0.06]),0.06]);

fprintf('np=%.2f\n',noise_power)
noise_th1 = 150*noise_power;
noise_th2 = 15*noise_power;

sflag = 0;
clip  = zeros(size(x));
wp = zeros(size(x));
chunks = zeros(size(x));

wp_buff_len = 10;
wp_buff = zeros(1,wp_buff_len);


for i = 2:(length(indice)-1)    
   fin = min(indice(i)+N1, length(x));
   window_power = sum(abs(x(indice(i):fin).^2));
   wp_buff = [wp_buff(2:end) window_power];
   if (sum(wp_buff) > noise_th1 && sflag==0)
       sflag = 1;
       clip(indice(i-wp_buff_len)) = sum(wp_buff);
       chunks(indice(i-wp_buff_len)) = 1;
   elseif (sum(wp_buff) < noise_th2 && sflag==1 )
        clip(min([indice(i)+N1, L1])) = sum(wp_buff);
        sflag = 0;
        chunks(min([indice(i)+N1, L1])) = 1;
   end
   wp(indice(i)) = window_power;
end

close all
figure
plot((1:length(x)), x)
hold on
yyaxis right
plot((1:length(x)), clip)
%plot((1:length(x))/fs, wp+eps)
%plot((1:length(x))/fs, chunks*80)

legend('signal','cl')
fprintf('th1=%.2f\n',noise_th1)
fprintf('th2=%.2f\n',noise_th2)

fprintf('MaxWP=%.2f\n',max(wp))
fprintf('Chunks=%d\n',length(find(chunks)))
%%

chidx = find(chunks);
figure
for i = 1:2:length(chidx)
   plot(x(chidx(i):chidx(i+1)))
   ylim([-1 1])
   soundsc(x(chidx(i):chidx(i+1)),fs)
   pause
end

close all
%% Save files
path = './palabras/';

audiowrite(filename,y,Fs)



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
