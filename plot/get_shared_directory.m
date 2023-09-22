function shared_directory = get_shared_directory

possible_directories = {...
    '/scratch/snormanh_lab/shared', ... % where everything is on BlueHive
    '/Users/svnh2/Desktop/projects/lab-shared', ... % Sam's local copy
    };

shared_directory = '';
for i = 1:length(possible_directories)
    if exist(possible_directories{i}, 'dir')
        shared_directory = possible_directories{i};
        return;
    end
end