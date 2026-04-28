% pathnew_matlab_central
%

% path-to-speech: starting directory definition
pathtospeech='C:\data\matlab_central_speech'

% paths to GUI toolkit
path(strcat(pathtospeech,'\gui_lite_2.6\GUI Lite v2.6'),path);

% paths to speech toolkit
path(strcat(pathtospeech,'\functions_lrr'),path);
path(strcat(pathtospeech,'\speech_files'),path);

% path to highpass filter mat files
path(strcat(pathtospeech,'\highpass_filter_signal'),path);

% path to cepstral coefficient training files
path(strcat(pathtospeech,'\VQ'),path);

% path to cepstral coefficients
path(strcat(pathtospeech,'\cepstral coefficients'),path);

% path to lrr isolated digit files set for training and testing
path(strcat(pathtospeech,'\isolated_digit_files\testing set'),path);
path(strcat(pathtospeech,'\isolated_digit_files\training set'),path);