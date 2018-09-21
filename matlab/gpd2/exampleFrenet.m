% 
% This example illustrates the process of calculating frenet frames for
% points sampled from a cloud.
% 
function exampleFrenet

    close all;
    
    % Both hand parameters and image parameters must be the same here as
    % they were during training
    handparams = handParameters();
    handparams.fw = 0.01;
    handparams.handOD = 0.12; % 0.09?
    handparams.handDepth = 0.06;
    handparams.handHeight = 0.02;

    % Get random point cloud
    p = clsPts();
    p = p.addCamPos([0;0;0]);
    p = p.loadFromFile(...
        '/home/baxter/data/active_sensing_verification/verified/both-2016-04-15-13-28.pcd', ...
        1,0);
    
    f = clsPtsFrenet(p);
    
    % Calculate surface normals using Matlab's built-in function
    f = f.setCloud(p);
    f = f.calcCloudNormals();

    figure;
    f.cloud.plot(); hold on;
    f.cloud.plotNormals(100); % plot a sampling of 100 normals
    title('surface normals');
    
    % Calculate frenet frames at 100 samples just as we would during grasp
    % candidate generation.
    f = f.subSample(100);
    f = f.evalBall(0.01);
    f = f.evalFrenetFrame();
             
    figure;
    f.plot(); hold on;
    f.plotNormals(100); % plot a sampling of 100 normals
    title('frenet frame normals');

end
