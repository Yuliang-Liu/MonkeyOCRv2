files = dir('../images/*.jpg');

for i=1:length(files)
    file = files(i);
    fpath = sprintf('%s/%s', file.folder, file.name);
    obj = jpeg_read(fpath);
    fname = sprintf('coef_arrays_%s.mat', files(i).name);
    coef_arrays = obj.coef_arrays;
    save(fname, 'coef_arrays');
end
    