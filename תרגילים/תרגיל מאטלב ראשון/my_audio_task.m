% --- Step 1: Load and Prepare ---

[y_full, Fs] = audioread('full_record.unknown'); % Use your exact file name

% Convert to mono (single channel) if it's stereo
if size(y_full, 2) == 2
    y_full = mean(y_full, 2);
end

disp('Full record loaded. Playing the entire track:');

% --- Step 2: Plot Full Waveform to Find Segments ---

% Create a time vector for the x-axis
t_full = (0:length(y_full)-1) / Fs;

% Plot the waveform
figure;
plot(t_full, y_full);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Full Recording - Find Your Split Points Here!');
grid on;
zoom on; % Lets you zoom in

% --- Step 3: Segment the Audio ---

% !!! --- YOU MUST CHANGE THESE VALUES --- !!!
% Look at your plot and enter the END time for each part
time_spkr1_ends = 1.86;   % EXAMPLE: Change this to your 1st split time
time_spkr2_ends = 3.87;  % EXAMPLE: Change this to your 2nd split time
% --- (The merged part is just everything after)

% Convert these times into MATLAB vector indices
idx1_start = 1;
idx1_end = round(time_spkr1_ends * Fs);

idx2_start = idx1_end + 1;
idx2_end = round(time_spkr2_ends * Fs);

idx_merge_start = idx2_end + 1;
idx_merge_end = length(y_full); % Goes to the very end

% Create the new variables by "slicing" the main vector
y1 = y_full(idx1_start : idx1_end);
y2 = y_full(idx2_start : idx2_end);
y_merge = y_full(idx_merge_start : idx_merge_end);

disp('Signal segmented into y1, y2, and y_merge.');

% --- Step 4: Spectrogram Analysis ---

% Define spectrogram parameters
win_length = 4096; % Long window for good FREQUENCY resolution
overlap = win_length / 2; % 50% overlap
nfft = win_length; % Number of FFT points

% Plot Spectrogram for Speaker 1
figure;
spectrogram(y1, win_length, overlap, nfft, Fs, 'yaxis');
title('Speaker 1 (Segmented)');

% Plot Spectrogram for Speaker 2
figure;
spectrogram(y2, win_length, overlap, nfft, Fs, 'yaxis');
title('Speaker 2 (Segmented)');

% Plot Spectrogram for the Merged recording
figure;
spectrogram(y_merge, win_length, overlap, nfft, Fs, 'yaxis');
title('Merged Recording (Segmented)');

% --- Step 5: Attempt Separation with Filtering ---

% !!! --- YOU MUST CHANGE THIS VALUE --- !!!
% Look at your spectrograms from Step 4.
% Pick a frequency (in Hz) that is BETWEEN the two speakers'
% main fundamental frequencies.
cutoff_freq = 175;
% ---

% Create a low-pass filter to TRY to isolate the lower voice
y_low_filtered = lowpass(y_merge, cutoff_freq, Fs);

% Create a high-pass filter to TRY to isolate the higher voice
y_high_filtered = highpass(y_merge, cutoff_freq, Fs);

% --- Step 6: Listen and View the Results ---

disp('Playing LOW-filtered sound...');
sound(y_low_filtered, Fs);
pause(length(y_low_filtered)/Fs + 1);

disp('Playing HIGH-filtered sound...');
sound(y_high_filtered, Fs);

% Plot the spectrograms of the filtered signals
figure;
subplot(2, 1, 1);
spectrogram(y_low_filtered, win_length, overlap, nfft, Fs, 'yaxis');
title(['Low-Pass Filtered at ' num2str(cutoff_freq) ' Hz']);

subplot(2, 1, 2);
spectrogram(y_high_filtered, win_length, overlap, nfft, Fs, 'yaxis');
title(['High-Pass Filtered at ' num2str(cutoff_freq) ' Hz']);

% השיטה של להפריד לפי תדר מסוים לא עובדת (מן הסתם בגלל שלדיבור אין תדר
% יחיד