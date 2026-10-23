import csv
from glob import glob
import os
import numpy as np


def slices(groups):
    """
    Generate slices to select
    groups of the given sizes
    within a list/matrix
    """
    i = 0
    for group in groups:
        yield i, i + group
        i += group

def load_image_list( out_dir ):

    img_files = []
    fd = open( out_dir + 'sample_images.csv' )
    for line in fd:
        files = [ fn.strip() for fn in line.split(',')[1:] if fn.strip() != '' ]
        img_files.extend( files )
    return img_files

def load_mask_list( out_dir ):

    mask_files = []
    fd = open( out_dir + 'sample_masks.csv' )
    for line in fd:
        mask_files.extend( [ fn.strip() for fn in line.split(',')[1:] if fn.strip() != '' ] )
    return mask_files

def load_sample_images( out_dir ):

    samples = {}
    fd = open( out_dir + 'sample_images.csv' )
    for line in fd:
        line = line.split(',')
        samples[line[0]] = [ fn.strip() for fn in line[1:] if fn.strip() != '' ]
    return samples

def load_labels( out_dir ):

    samples = []
    labels = []
    #d = np.loadtxt( out_dir+'labels.csv', dtype=str, delimiter=',' )
    #d = np.array(d)
    d = []
    with open( out_dir+'labels.csv', 'r' ) as csvfile:
        reader = csv.reader( csvfile )
        for row in reader:
            d.append( row )
    #print([len(di) for di in d])
    d = np.vstack(d)
    samples = d[1:,0]
    cats = d[0,1:]
    labels = d[1:,1:]
    return list(samples),cats,labels

def load_cv_files( out_dir, samples, cv_fold_files ):

    cv_files = sorted(list(glob( out_dir + cv_fold_files )))
    idx_train_test = []
    for fn in cv_files:
        print(fn)
        f = np.loadtxt( fn, dtype=str, delimiter=',' )
        idx_train = np.where(f[:,1]=='train')[0]
        idx_test = np.where(f[:,1]=='test')[0]
        name_train = f[idx_train,0]
        name_test = f[idx_test,0]
        idx_train = np.array([ np.where(samples==name)[0] for name in name_train ]).flatten()
        idx_test = np.array([ np.where(samples==name)[0] for name in name_test ]).flatten()
        idx_train_test.append( [idx_train,idx_test] )
    return idx_train_test

def load_feats(out_dir, sample_images, model_name, layer, pool_size, instance_size, instance_stride, mi_type):
    # read in CNN features
    feats = {}
    missing_file_count = 0
    skipped_sample_count = 0
    for sample, imagelist in sample_images.items():
        sample_feats = []
        for fn in imagelist:
            feat_fn = out_dir + fn[:fn.rfind('.')] + '_' + model_name + '-' + layer
            if pool_size is not None:
                feat_fn += '_p' + str(pool_size)
            if instance_size is not None:
                feat_fn += '_i' + str(instance_size)
            if instance_stride is not None:
                feat_fn += '-' + str(instance_stride)
            feat_fn += '.npy'
            if not os.path.exists(feat_fn):
                missing_file_count += 1
                print('Warning: missing feature file, skip image: %s' % feat_fn)
                continue
            try:
                feat = np.load(feat_fn)
            except Exception as exc:
                missing_file_count += 1
                print('Warning: failed to load feature file, skip image: %s (%s)' % (feat_fn, str(exc)))
                continue
            if len(feat) == 0:
                continue
            sample_feats.append(feat)

        if len(sample_feats) == 0:
            skipped_sample_count += 1
            print('Warning: no valid feature file for sample, skip sample: %s' % sample)
            continue

        feats[sample] = np.concatenate(sample_feats, axis=0)
        if len(feats[sample].shape) == 1:
            feats[sample] = feats[sample].reshape((1, len(feats[sample])))

        # compute mean if needed
        if mi_type is None or mi_type.lower() == 'none':
            if len(feats[sample].shape) > 1:
                feats[sample] = feats[sample].mean(axis=0)

    if missing_file_count > 0:
        print('Warning: skipped %d missing/corrupt feature files under %s' % (missing_file_count, out_dir))
    if skipped_sample_count > 0:
        print('Warning: skipped %d samples with no valid features under %s' % (skipped_sample_count, out_dir))

    return feats

def clean_cats_labels(cats, labels, categories):
    if categories is None:
        # get labels for all categories
        label_names = []
        new_labels = np.zeros(labels.shape, dtype='int')
        for c, cat in enumerate(cats):
            ln = np.unique([l[c] for l in labels])
            ln.sort()
            ln = list(ln)
            label_names.append(ln)
            new_labels[:, c] = [ln.index(l) for l in labels[:, c]]
        labels = new_labels
    else:
        # get labels for list of categories
        label_names = []
        categories = categories.split(',')
        new_labels = np.zeros((labels.shape[0], len(categories)), dtype='int')
        for i, cat in enumerate(categories):
            c = np.where(cats == cat)[0][0]
            ln = np.unique([l[c] for l in labels])
            ln.sort()
            ln = list(ln)
            if '' in ln:
                del ln[ln.index('')]
            label_names.append(ln)
            new_labels[:, i] = np.array([ln.index(l) if l in ln else -1 for l in labels[:, c]])
        labels = new_labels
        cats = categories

        return cats, labels, label_names

def load_multi_data(dirs, model_name, layer, pool_size, instance_size, instance_stride, mi_type, categories, return_groups=False):
    multi_feats = {}
    multi_sample_images = {}
    multi_samples = []
    multi_labels = None
    if return_groups:
        groups = []
    for dir in dirs:
        # load feats
        sample_images = load_sample_images(dir)
        feats = load_feats(dir, sample_images, model_name, layer, pool_size, instance_size, instance_stride, mi_type)
        multi_feats.update(feats)
        # load rest of data
        samples, cats, labels = load_labels(dir)
        cats, labels, label_names = clean_cats_labels(cats, labels, categories)
        multi_sample_images.update(sample_images)
        multi_samples += list(samples)
        if multi_labels is None:
            multi_labels = labels
        else:
            multi_labels = np.concatenate([multi_labels, labels], axis=0)
        # store group counts
        if return_groups:
            groups.append(len(sample_images))
    if return_groups:
        return multi_feats, multi_samples, cats, multi_labels, label_names, groups

    return multi_feats, multi_samples, cats, multi_labels, label_names
