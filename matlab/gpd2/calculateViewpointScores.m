% 
% Calculate and display expected score, accuracy, and density of positives
% as a function of viewpoint. Results on rectangular objects appear to show
% that viewpoint can have a singificant effect.
% 
% Saves off viewpoint/score info to ./data/viewpointData.mat
% 
function calculateViewpointScores()

    close all;
    
    categoryroot = './data/MATfiles/';
    folderset = {'advil'};

    % Import data from *_Images.mat files into clsLearning::l
    useParfor = 0;
    tc = clsTrainCaffe(categoryroot,folderset,useParfor,'');
    l = clsLearning();
    maxNumPerObject = 10000;
    l = tc.loadClsLearning(l,maxNumPerObject);

    % We are going to characterize positives, so let's prune negatives
    l = l.prune(find(l.labels));
    
    % Direction from which each hand was viewed
    unitSourceDirections = l.imageCamPos();
    unitSourceDirections = unitSourceDirections .* repmat(1./sqrt(sum(unitSourceDirections.^2,1)),3,1);
    
    % Ground truth labels
    labelsGT = l.labels;
    labelsGT_pm1 = 2*l.labels - 1;

    % 3dplot of viewing directions relative to hands
    figure;
    temp = unitSourceDirections(:,l.labels == 1);
    plot3(temp(1,:),temp(2,:),temp(3,:),'ro');
    hold on;
    temp = unitSourceDirections(:,l.labels == 0);
    plot3(temp(1,:),temp(2,:),temp(3,:),'bo');
    title('colored by ground truth label (red positive, blue negative)');
    xlabel('x');
    ylabel('y');
    zlabel('z');
    axis equal;

    % calculate scores, accuracy
    scores = l.getScores('./deploy_oldversion.prototxt', './data/CAFFEfiles/lenet_iter_1000.caffemodel', 0);
    scores = scores(:,2)' - scores(:,1)';
    correctPredictions = l.labels == (scores>0);
    accuracy = sum(correctPredictions) / size(scores,2);

    % plot histogram of scores
    figure;
    histogram(scores);
    
    % Calculate expected scores, accuracy, and positive frequency as a
    % function of viewing angle.
%     thetas = pi/2:0.05:1.5*pi;
%     thetas = -pi:0.05:pi;
    thetas = 0:0.05:2*pi;
    phis = -pi/2:0.05:pi/2;
    fieldSize = 0.2;
    for iterPhi = 1:size(phis,2)
        for iterTheta = 1:size(thetas,2)
            
            phi = phis(iterPhi);
            theta = thetas(iterTheta);
            
            xx = cos(theta) * cos(phi);
            yy = sin(theta) * cos(phi);
            zz = sin(phi);
            sp = [xx;yy;zz];

            delta = unitSourceDirections - repmat(sp,1,l.num());
            dist = sqrt(sum(delta.^2,1));

            % weight points using gaussian kernels
            weights = exp(-dist.^2/fieldSize^2);
%             weights = weights / sum(weights);
            
            scoreSphere(iterPhi,iterTheta) = sum(scores .* weights); % expected score
            accuracySphere(iterPhi,iterTheta) = sum(correctPredictions.* weights); % expected accuracy
            labelsGTSphere(iterPhi,iterTheta) = sum(labelsGT_pm1 .* weights); % expected frequency of positives
        end
    end
    
    
    % Display expected scores, accuracy and positive frequency
    [Phis, Thetas] = meshgrid(thetas,phis);
    figure;
    surf(Phis,Thetas,scoreSphere);
    xlabel('theta');
    ylabel('phi');
    axis([0 2*pi -pi/2 pi/2 -inf inf]);
%     axis([-pi pi -pi/2 pi/2 -inf inf]);
    title('scores');

    figure;
    surf(Phis,Thetas,labelsGTSphere);
    xlabel('theta');
    ylabel('phi');
    axis([0 2*pi -pi/2 pi/2 -inf inf]);
%     axis([-pi pi -pi/2 pi/2 -inf inf]);
    title('density of positives');
    
    figure;
    surf(Phis,Thetas,accuracySphere);
    xlabel('theta');
    ylabel('phi');
    axis([0 2*pi -pi/2 pi/2 -inf inf]);
%     axis([pi/2 1.5*pi -pi/2 pi/2 -inf inf]);
    title('accuracy');
    
    
    % Save it off so we can use these expectations elsewhere
    save('./data/viewpointData.mat','unitSourceDirections','labelsGT','-v7.3');
end