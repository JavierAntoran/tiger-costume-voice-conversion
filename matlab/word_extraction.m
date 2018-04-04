
word_path = 'palabras/';

prefix = 'j';

phrase_idx = kron(1:6,[1 1 1])';
phrase_rep = reshape(kron([1:3]',ones(1,6)),1,6*3)';

files = sprintfc('j%d%d.wav',[phrase_idx,phrase_rep]);

for fileidx = 1:numel(files)
    [x,fs] = audioread(files{fileidx});

    N= 512;
    W = hamming(N);

    x = x(:,1);

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
    files{fileidx}
    fprintf('th1=%.2f\n',noise_th1)
    fprintf('th2=%.2f\n',noise_th2)
    fprintf('Chunks=%d\n',length(find(chunks)))
    chidx = find(chunks);
    word_idx = 1;
    for i = 1:2:length(chidx)
        filename = strcat(word_path,prefix,num2str(phrase_idx(fileidx)),num2str(phrase_rep(fileidx)),'_',num2str(word_idx),'.wav');
        audiowrite(filename,x(chidx(i):chidx(i+1)),fs)
        word_idx = word_idx+1;
    end

    
end