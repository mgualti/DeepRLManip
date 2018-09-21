function AggregateResults()

    %% Parameters
    
    saveFilePrefix = '2018-05-02';
    saveFilePostfixes = {'AverageReturn','AverageReward'};
    figsToSave = [];

    %% Load

    close('all');

    files = dir('../*results.mat');
    if isempty(files), return; end
    data = cell(length(files));
    for idx=1:length(files)
        data{idx} = load([files(idx).folder '/' files(idx).name]);
    end
    figs = [];
    
    %% Best Run
    
    bestReturn = -Inf;
    bestRunIdx = -1;
    for idx=1:length(data)
        r = mean(data{idx}.avgReturn(end-5:end));
        if r > bestReturn
            bestReturn = r;
            bestRunIdx = idx;
        end
    end
    disp(['Best run was ' files(bestRunIdx).name ' with return ' num2str(bestReturn)]);
    
    %% Plot Average Return
    
    figs = [figs, figure]; hold('on')
    for idx=1:length(data)
        plot(data{idx}.avgReturn, 'linewidth', 1);
    end
    grid('on'); xlabel('Training Round'); ylabel('Reward');
    title('Average Return');
    
    %% Plot Grasping and Placing Results
    
    figs = [figs, figure]; hold('on')
    for idx=1:length(data)
        plot(data{idx}.avgGraspReward, 'color', 'r', 'linewidth', 1.5);
        plot(data{idx}.avgPlaceReward, 'color', 'b', 'linewidth', 1.5);
    end
    l = legend('Grasp', 'Place');
    l.Location = 'best'; l.FontSize=14; l.FontWeight = 'bold'; l.Box='off';
    legend('boxoff'); grid('on'); set(gca, 'FontSize', 14)
    xlabel('Training Round', 'fontweight', 'bold', 'fontsize', 14);
    ylabel('Average Reward', 'fontweight', 'bold', 'fontsize', 14);
    %title('Average Reward');
    
    %% Saving Images
    
    for idx=1:length(figsToSave)
        saveas(figs(figsToSave(idx)), ['../../notebook/figures-2/' ...
            saveFilePrefix '-' saveFilePostfixes{figsToSave(idx)} '.png']);
    end
    
end