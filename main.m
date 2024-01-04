system('conda activate base');
path ="E:\code\unet1\unet";
name = "one_1.jpg";
command = sprintf('python test.py %s %s',name,path);
[~,cmdout] = system(command);