% Struct for hand parameters. These parameters have a large effect on which
% hands are discovered by clsHoodHands::findHands
classdef handParameters
    properties
        fw; % width of finger
        handDepth; % depth of hand
        handOD; % outer diameter of the hand
        handHeight; % height of hand
    end    
end