files = dir('../images/*.jpg');

for i=1:length(files)
    file = files(i);
    fpath = sprintf('%s/%s', file.folder, file.name);
    obj = jpeg_read(fpath);
    fname = sprintf('nnz_%s.mat', files(i).name);
    nnz_ac = 0;
    for j=1:length(obj.coef_arrays)
        coef  = obj.coef_arrays{j};
        nnz_ac = nnz_ac + (nnz(coef) - nnz(coef(1:8:end,1:8:end)));
    end    
    save(fname, 'nnz_ac');
end