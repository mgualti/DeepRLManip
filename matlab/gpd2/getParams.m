% 
% Set handparams and imageparams here
% 
function [handparams, imageparams] = getParams()
    
    % this corresponds to the actual hand (Robotiq 85)
    %handparams = handParameters();
    %handparams.fw = 0.01;
    %handparams.handOD = 0.105;
    %handparams.handDepth = 0.06;
    %handparams.handHeight = 0.02;
    
    % same as hvs parameters
    handparams = handParameters();
    handparams.fw = 0.01;
    handparams.handOD = 0.105;
    handparams.handDepth = 0.075;
    handparams.handHeight = 0.01;
    
    % this corresponds to the hand used for training
    imageparams = imageParameters();
    imageparams.imageOD = 0.10;
    imageparams.imageDepth = 0.06;
    imageparams.imageHeight = 0.02;
    imageparams.imageSize = 60;

end
