% This is the first file you run when doing training for a new object.
% 
% Creates the following files in <categoryroot>:
% *_allclouds.mat (contains clouds extracted from bigbird aligned into a single reference frame)
% *_clouds4Learning.mat (contains clouds combined into a single clsPts)
% *_LabeledGraspCandidates.mat (contains labeled grasp candidates)
% *_Images.mat (contains images paired with labels for training)
% 
% You need to modify the following:
% categoryroot -> point to a directory where you would like to store the
%                 .MAT files.
% bbroot -> point to the bigbird subdir
% folderset -> point to at least one existing subdir under <category> root.
%              Inside this subdir, there must be a file called
%              category_object_list.txt that contains a list of all objects
%              in this subdir for which you want to extract data. No <cr>
%              on the last line. See example file in gpdinc directory.
%
% When run on advil_liqui_gels, this file generates a balanced training
% sets of approx 1k exemplars. To increase this, just increase numSamples.
% More training data can only help...
%
function createImagesMugs

    close all;
    
    % These DEBUG parameters turn on/off unit tests for the functions
    % indicated. If turned on, it will produce plots that you can use to
    % verify that the code is functioning correctly.
    DEBUG = {};
    %DEBUG = [DEBUG, {'clsPtsHands.getGraspCandidates'}];
    %DEBUG = [DEBUG, {'clsPtsHands.calculateLabels'}];
    %DEBUG = [DEBUG, {'clsPtsHands.calculateImage'}];
    %DEBUG = [DEBUG, {'clsPtsHands.calculateOcclusion'}];
    
    % *** All your .MAT files get saved to a path off <categoryroot>
    categoryroot = './data/MATfiles/';
%     categoryroot = '~/projects/object_datasets/bb_onesource/3dnet/';

    % *** <bbroot> is where your BB or 3dnet data resides
    bbroot = '/home/mgualti/Data/3DNet/Cat10_ModelDatabase/'; % this is for 3DNet
    
    % *** <folderset> is a set of directories off <categoryroot>. Each
    % directory must contain a txt file called category_object_list.txt
    % that contains the name of each object for which to create images.
    % It is intended that each directory contains a set of objects that
    % belong to a different category.
    folderset = {'mug'};

    % *** Generate *_allclouds.mat and *_clouds4Learning.mat for each
    % object. Each *_clouds4Learning.mat file contains one clsPts strucutre
    % containing object views and one clsPts structure containing the
    % ground truth mesh. 
    allCamSources = 0:3:357;
    voxresolution = 0.002;
    curvAxisProb = 0.33;

    bb = clsBigBird(bbroot, categoryroot, allCamSources, voxresolution);
    bb.extractCloudsObjectSet3DNet(folderset, 0.3, [0.055 0.090]); % 3DNET; baseline, desiredWidthRange; desiredWidthRange is on the large side because we typically overestimate object size slightly

    useParfor = 1; % set this to 0 to run serially
    trainCaffe = clsTrainCaffe(categoryroot, folderset, useParfor, DEBUG);

    [handparams, imageparams] = getParams();
    
    % Find grasp candidates, label them, balance the dataset, and save off
    % to *_LabeledGraspCandidates.mat files.
    numSamples = 500;
    trainCaffe.calcLabeledGrasps(handparams, 20, numSamples, 1, curvAxisProb);

    % For each grasp candidate saved to a *_LabeledGraspCandidates.mat
    % file, calculate the image and save off to *_Images.mat file.
    % These imageparams contain the interior of the gripper (standard).
    trainCaffe.calcOcclusionImages(imageparams);

end

