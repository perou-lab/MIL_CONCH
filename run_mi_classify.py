import os
import sys
import argparse
import csv
import warnings
import numpy as np
import sklearn.metrics
import sksurv.metrics
from joblib import dump, load

import util
from linear_classifier import LinearClassifier
from sil import SIL


class ResultsReport:
    def __init__(self, label_names=None):
        self.res = {}
        self.label_names = label_names
        self.per_class = {}

    def add(self, metric, result):
        if metric not in self.res:
            self.res[metric] = []
        self.res[metric].append(result)

    def print_summary(self, metric=None):
        if metric is None:
            if 'acc' not in self.res or not self.res['acc']:
                print("No results to summarize.")
                return

            acc_arr = np.array(self.res['acc'], dtype=float)
            valid_acc = np.sum(~np.isnan(acc_arr))
            acc_mean = np.nanmean(acc_arr)
            acc_sem = np.nanstd(acc_arr) / np.sqrt(valid_acc) if valid_acc > 0 else float('nan')

            auc_arr = np.array(self.res['auc'], dtype=float) if 'auc' in self.res else np.array([])
            valid_auc = np.sum(~np.isnan(auc_arr))
            has_auc = valid_auc > 0
            if has_auc:
                auc_mean = np.nanmean(auc_arr)
                auc_sem = np.nanstd(auc_arr) / np.sqrt(valid_auc)
            else:
                auc_mean = float('nan')
                auc_sem = float('nan')

            has_binary_metrics = (
                'sensitivity' in self.res
                and 'specificity' in self.res
                and len(self.res['sensitivity']) > 0
                and len(self.res['specificity']) > 0
            )
            if has_binary_metrics:
                sens_arr = np.array(self.res['sensitivity'], dtype=float)
                spec_arr = np.array(self.res['specificity'], dtype=float)
                valid_sens = np.sum(~np.isnan(sens_arr))
                valid_spec = np.sum(~np.isnan(spec_arr))
                if valid_sens > 0 and valid_spec > 0:
                    sens_mean = np.nanmean(sens_arr)
                    sens_sem = np.nanstd(sens_arr) / np.sqrt(valid_sens)
                    spec_mean = np.nanmean(spec_arr)
                    spec_sem = np.nanstd(spec_arr) / np.sqrt(valid_spec)
                    print(
                        """
                Accuracy Sensitivity Specificity AUC
                {:.3f} ({:.3f}) & {:.3f} ({:.3f}) & {:.3f} ({:.3f}) & {:.3f} ({:.3f})
                """.format(
                            acc_mean,
                            acc_sem,
                            sens_mean,
                            sens_sem,
                            spec_mean,
                            spec_sem,
                            auc_mean,
                            auc_sem,
                        )
                    )
                else:
                    has_binary_metrics = False
            if not has_binary_metrics:
                auc_line = ""
                if has_auc:
                    auc_line = f"{auc_mean:.3f} ({auc_sem:.3f})"
                else:
                    auc_line = "N/A"
                print(
                    f"""
                Accuracy AUC
                {acc_mean:.3f} ({acc_sem:.3f}) & {auc_line}
                """
                )
            if 'confusion' in self.res:
                self.print_summary('confusion')

            if self.per_class:
                print("Per-class performance:")
                for cls, stats in self.per_class.items():
                    if not stats:
                        continue
                    acc_vals = stats.get('accuracy', [])
                    if acc_vals:
                        acc_arr = np.array(acc_vals, dtype=float)
                        valid = np.sum(~np.isnan(acc_arr))
                        if valid > 0:
                            acc_mean = np.nanmean(acc_arr)
                            acc_sem = np.nanstd(acc_arr) / np.sqrt(valid)
                            print(f"  {cls} Accuracy: {acc_mean:.3f} ({acc_sem:.3f})")
                        else:
                            print(f"  {cls} Accuracy: N/A")
                    sens_vals = stats.get('sensitivity', [])
                    if sens_vals:
                        sens_arr = np.array(sens_vals, dtype=float)
                        valid = np.sum(~np.isnan(sens_arr))
                        if valid > 0:
                            sens_mean = np.nanmean(sens_arr)
                            sens_sem = np.nanstd(sens_arr) / np.sqrt(valid)
                            print(f"  {cls} Sensitivity: {sens_mean:.3f} ({sens_sem:.3f})")
                        else:
                            print(f"  {cls} Sensitivity: N/A")
                    spec_vals = stats.get('specificity', [])
                    if spec_vals:
                        spec_arr = np.array(spec_vals, dtype=float)
                        valid = np.sum(~np.isnan(spec_arr))
                        if valid > 0:
                            spec_mean = np.nanmean(spec_arr)
                            spec_sem = np.nanstd(spec_arr) / np.sqrt(valid)
                            print(f"  {cls} Specificity: {spec_mean:.3f} ({spec_sem:.3f})")
                        else:
                            print(f"  {cls} Specificity: N/A")
                    auc_vals = stats.get('auc', [])
                    if auc_vals:
                        auc_arr = np.array(auc_vals, dtype=float)
                        valid = np.sum(~np.isnan(auc_arr))
                        if valid > 0:
                            auc_mean = np.nanmean(auc_arr)
                            auc_sem = np.nanstd(auc_arr) / np.sqrt(valid)
                            print(f"  {cls} AUC: {auc_mean:.3f} ({auc_sem:.3f})")
                        else:
                            print(f"  {cls} AUC: N/A")

            if c_index:
                self.print_summary('c_index_continuous')
                self.print_summary('c_index_binary')
                # print(f"C-index (continuous): {self.res['c_index_continuous']}")
                # print(f"C-index (binary): {self.res['c_index_binary']}")
            # for metric in sorted(self.res.keys()):
            # for metric in self.res.keys():
            #     if metric != 'confusion':
            #         self.print_summary(metric)
            return
        if metric != 'confusion':
            mean = np.mean(self.res[metric])
            std = np.std(self.res[metric])
            ste = std / np.sqrt(len(self.res[metric]))
            print('%s %f %f %f' % (metric, mean, std, ste))
        # def print_summary(self, metric=None):
        #     if metric is None:
        #         print(f"""
        #         Accuracy Sensitivity Specificity AUC
        #         {np.mean(self.res['acc'])},{np.mean(self.res['sensitivity'])},{np.mean(self.res['specificity'])},{np.mean(self.res['auc'])}
        #         """)
        #         if c_index:
        #             self.print_summary('c_index_continuous')
        #             self.print_summary('c_index_binary')
        #             # print(f"C-index (continuous): {self.res['c_index_continuous']}")
        #             # print(f"C-index (binary): {self.res['c_index_binary']}")
        #         # for metric in sorted(self.res.keys()):
        #         # for metric in self.res.keys():
        #         #     if metric != 'confusion':
        #         #         self.print_summary(metric)
        #         self.print_summary('confusion')
        #         return
        #     if metric != 'confusion':
        #         mean = np.mean(self.res[metric])
        #         std = np.std(self.res[metric])
        #         ste = std / np.sqrt(len(self.res[metric]) - 1)
        #         print('%s %f %f %f' % (metric, mean, std, ste))

        else:
            print('confusion')
            print(('%s ' * len(self.label_names)) % tuple(self.label_names))
            print(sum(self.res['confusion']))
            print('Negative / Positive')
            print(np.sum(sum(self.res['confusion']), axis=1))


def _mkdir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def _write_tsv(path, fieldnames, rows):
    _mkdir(os.path.dirname(path))
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def _append_tsv(path, fieldnames, row):
    _mkdir(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, 'a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _label_name(label_names, label_value):
    try:
        label_value = int(label_value)
        if 0 <= label_value < len(label_names):
            return label_names[label_value]
    except Exception:
        pass
    return str(label_value)


def _binary_count(values, positive_value=1):
    values = np.asarray(values)
    positive = int(np.sum(values == positive_value))
    negative = int(len(values) - positive)
    rate = positive / len(values) if len(values) else float('nan')
    return positive, negative, rate


def _split_tag(train_size, test_size):
    return 'train{:02d}_test{:02d}'.format(int(round(float(train_size) * 100)), int(round(float(test_size) * 100)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and test MI classifier.')
    parser.add_argument('--out_dir', '-o', required=True, help='output directory')
    parser.add_argument('--model', '-m', required=True, help='CNN model')
    parser.add_argument('--layer', '-l', help='CNN layer (default: end)')
    parser.add_argument('--instance-size', help='instance size')
    parser.add_argument('--instance-stride', help='instance stride')
    parser.add_argument('--pool-size', '-p', help='mean pooling size')
    parser.add_argument('--cat', help='label categories to train (comma separated); default: all')
    parser.add_argument('--calibrate', action='store_true', help='calibrate classifier')
    parser.add_argument('--metric',
                        help='metric to optimize during parameter search (accuracy, balanced_accuracy, roc_auc); default: accuracy')
    parser.add_argument('--classifier', '-c', help='classifier (logistic, svm, dwd); default: svm')
    parser.add_argument('--kernel', help='SVM kernel; default: linear')
    parser.add_argument('--mi',
                        help='MI type (none, median, max, quantile, quantile_mean, quantile_pca, quantile_mean_pca, quantile_opc); default: none (compute mean across images)')
    parser.add_argument('--agg', help='Aggregate type (svm, dwd); default: svm')
    parser.add_argument('--quantiles', '-q', help='Number of quantiles; default: 16')
    parser.add_argument('--sample-weight', help='Weight samples by classification category and this one')
    parser.add_argument('--group', help='Class groups for reporting results')
    parser.add_argument('--cv-fold-files', help='cross-validation fold files')
    parser.add_argument('--cv-folds', help='cross-validation folds; default: 5 when no other split mode is specified')
    parser.add_argument('--cv-lno', help='cross-validation leave n out')
    parser.add_argument('--independent-cv', action='store_true',
                        help='for multiple categories, construct CV folds independently for each category')
    parser.add_argument('--split-mode', choices=['cv', 'stratified-holdout'], default='cv',
                        help='dataset split mode; default: cv')
    parser.add_argument('--train-size', help='training fraction for --split-mode stratified-holdout')
    parser.add_argument('--test-size', help='test fraction for --split-mode stratified-holdout')
    parser.add_argument('--n-repeats', help='number of independent stratified holdout repeats; default: 100')
    parser.add_argument('--repeat-start', help='for stratified-holdout, first 1-based repeat index to run; default: 1')
    parser.add_argument('--repeat-end', help='for stratified-holdout, last 1-based repeat index to run; default: n-repeats')
    parser.add_argument('--repeat-output-dir', help='base directory for repeated holdout models, splits, predictions, and metrics')
    parser.add_argument('--random-state', help='random state for splitting datasets')
    parser.add_argument('--save-first', action='store_true', help='save first trained classifier and datasets')
    parser.add_argument('--save-idx', action='store_true', help='save indices of first trained classifier and datasets')
    parser.add_argument('--c-index', action='store_true', help='c-index for survival analysis')
    parser.add_argument('--n-jobs', help='number of parallel threads')
    parser.add_argument('--n-components', help='number of principal components')
    parser.add_argument('--save-train', action='store_false', help='save trained classifier')
    parser.add_argument('--load-train', action='store_false', help='load trained classifier')
    args = parser.parse_args()
    out_dir = args.out_dir
    if len(out_dir) > 1 and out_dir[-1] != '/':
        out_dir += '/'
    model_name = args.model
    layer = args.layer if args.layer is not None else 'end'
    instance_size = args.instance_size
    instance_stride = args.instance_stride
    pool_size = args.pool_size
    categories = args.cat
    metric = args.metric
    calibrate = args.calibrate
    classifier = args.classifier
    kernel = args.kernel
    mi_type = args.mi
    agg_type = args.agg
    quantiles = args.quantiles
    sample_weight = args.sample_weight
    group = args.group
    cv_fold_files = args.cv_fold_files
    cv_folds = args.cv_folds
    cv_lno = args.cv_lno
    independent_cv = args.independent_cv
    split_mode = args.split_mode
    train_size = args.train_size
    test_size = args.test_size
    n_repeats = args.n_repeats
    repeat_start = args.repeat_start
    repeat_end = args.repeat_end
    repeat_output_dir = args.repeat_output_dir
    random_state = args.random_state
    save_first = args.save_first
    save_idx = args.save_idx
    c_index = args.c_index
    n_jobs = args.n_jobs
    n_components = args.n_components
    save_train = args.save_train
    load_train = args.load_train

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    if calibrate is None:
        calibrate = False
    else:
        calibrate = bool(calibrate)
        print(f'Calibrate: {calibrate}')

    if classifier is None:
        classifier = 'svm'

    if agg_type is None:
        agg_type = 'svm'

    if quantiles is None:
        quantiles = 16

    if save_first is None:
        save_first = False
    else:
        save_first = bool(save_first)

    if save_idx is None:
        save_idx = False
    else:
        save_idx = bool(save_idx)

    if c_index is None:
        c_index = False
    else:
        c_index = bool(c_index)

    if n_jobs is not None:
        n_jobs = int(n_jobs)

    if random_state is None:
        random_state = 111
    else:
        random_state = int(random_state)

    if n_repeats is None:
        n_repeats = 100
    else:
        n_repeats = int(n_repeats)
    repeat_start = 1 if repeat_start is None else int(repeat_start)
    repeat_end = n_repeats if repeat_end is None else int(repeat_end)

    if train_size is not None:
        train_size = float(train_size)
    if test_size is not None:
        test_size = float(test_size)
    if split_mode == 'stratified-holdout':
        if train_size is None and test_size is None:
            test_size = 0.2
            train_size = 0.8
        elif train_size is None:
            train_size = 1.0 - float(test_size)
        elif test_size is None:
            test_size = 1.0 - float(train_size)
        if train_size <= 0 or test_size <= 0 or train_size + test_size > 1.0 + 1e-8:
            raise ValueError('For stratified-holdout, train-size and test-size must be positive and sum to <= 1.')
        if repeat_start < 1 or repeat_end < repeat_start or repeat_end > n_repeats:
            raise ValueError('--repeat-start/--repeat-end must satisfy 1 <= start <= end <= n_repeats')
        if repeat_output_dir is None:
            repeat_output_dir = os.path.join(out_dir, 'repeated_holdout_results')

    if n_components is not None:
        n_components = int(n_components)

    if save_train is None:
        save_train = True
    else:
        save_train = bool(save_train)

    # if load_train is None:
    #     load_train = False
    # else:
    #     load_train = bool(load_train)
    load_train = False

    # load filenames and labels
    sample_images = util.load_sample_images(out_dir)
    samples, cats, labels = util.load_labels(out_dir)

    # TODO: only keep labels with corresponding feats

    if sample_weight is not None:
        # get labels for sample_weight category
        c = np.where(cats == sample_weight)[0][0]
        ln = np.unique([l[c] for l in labels])
        ln.sort()
        ln = list(ln)
        if '' in ln:
            del ln[ln.index('')]
        label_names_sw = ln
        labels_sw = np.array([ln.index(l) if l in ln else -1 for l in labels[:, c]])
    if group is not None:
        # get labels for group category
        if group == sample_weight:
            label_names_group = label_names_sw
            labels_group = labels_sw
        else:
            c = np.where(cats == group)[0][0]
            ln = np.unique([l[c] for l in labels])
            ln.sort()
            ln = list(ln)
            if '' in ln:
                del ln[ln.index('')]
            label_names_group = ln
            labels_group = np.array([ln.index(l) if l in ln else -1 for l in labels[:, c]])
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

    # read in CNN features
    feats = {}
    for sample, imagelist in sample_images.items():
        feats[sample] = []
        for fn in imagelist:
            feat_fn = out_dir + fn[:fn.rfind('.')] + '_' + model_name + '-' + layer
            if pool_size is not None:
                feat_fn += '_p' + str(pool_size)
            if instance_size is not None:
                feat_fn += '_i' + str(instance_size)
            if instance_stride is not None:
                feat_fn += '-' + str(instance_stride)
            feat_fn += '.npy'
            feat = np.load(feat_fn, allow_pickle=True)
            if isinstance(feat, np.ndarray) and feat.dtype == object:
                if len(feat) == 0:
                    continue
                feat = np.vstack([np.asarray(f, dtype=np.float32) for f in feat])
            else:
                feat = np.asarray(feat, dtype=np.float32)
            if len(feat) == 0:
                continue
            feats[sample].append(feat)

        feats[sample] = np.concatenate(feats[sample], axis=0)
        if len(feats[sample].shape) == 1:
            feats[sample] = feats[sample].reshape((1, len(feats[sample])))

        # compute mean if needed
        if mi_type is None or mi_type.lower() == 'none':
            if len(feats[sample].shape) > 1:
                feats[sample] = feats[sample].mean(axis=0)

    # build train/test sets
    idx = np.arange(len(samples))
    if cv_fold_files is not None:
        idx_train_test = util.load_cv_files(out_dir, samples, cv_fold_files)
    elif split_mode == 'stratified-holdout':
        pass
    elif cv_folds is not None or cv_lno is not None:
        if cv_folds is not None:
            cv_folds = int(cv_folds)
        else:
            cv_lno = int(cv_lno)
            if cv_folds is None:
                cv_folds = len(samples) // cv_lno
    else:
        cv_folds = 5

    options = {}
    if kernel is not None:
        options['kernel'] = kernel
    else:
        options['kernel'] = 'linear'
    if classifier is not None:
        options['classifier'] = classifier
    if mi_type is not None:
        options['predict_type'] = mi_type
        if agg_type is not None:
            options['agg_type'] = agg_type
    if metric is not None:
        options['metric'] = metric
    if n_components is not None:
        options['n_components'] = n_components

    for c, cat_name in enumerate(cats):
        idx_valid = [i for i in idx if (labels[i, c] != -1) & (samples[i] in feats)]
        labels_subset = labels[idx_valid]

        if split_mode == 'stratified-holdout':
            if len(label_names) == 1:
                stratify_labels = labels_subset[:, c]
            else:
                stratify_labels = np.array(labels_subset[:, c])
                p = 1
                for i in range(labels_subset.shape[1]):
                    stratify_labels += labels_subset[:, i] * p
                    p *= len(label_names[i])
            skf = sklearn.model_selection.StratifiedShuffleSplit(
                n_splits=n_repeats, train_size=train_size, test_size=test_size, random_state=random_state
            )
            idx_train_test = list(skf.split(idx_valid, stratify_labels))
        elif cv_fold_files is not None:
            pass
        elif len(label_names) == 1 or independent_cv:
            if cv_lno == 1:
                skf = sklearn.model_selection.LeaveOneOut()
            else:
                skf = sklearn.model_selection.StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                                              random_state=random_state)
            idx_train_test = list(skf.split(idx_valid, labels_subset[:, c]))
        else:
            # merge label categories to do stratified folds
            skf = sklearn.model_selection.StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            la_all = np.array(labels_subset[:, c])
            p = 1
            for i in range(labels_subset.shape[1]):
                la_all += labels_subset[:, i] * p
                p *= len(label_names[i])
            idx_train_test = list(skf.split(idx_valid, la_all))

        print(cat_name)
        res = ResultsReport(label_names[c])
        if split_mode == 'stratified-holdout':
            split_name_for_metrics = _split_tag(train_size, test_size)
            metric_path_for_reset = os.path.join(
                repeat_output_dir, 'metrics', cat_name, split_name_for_metrics + '_repeat_metrics.tsv'
            )
            if repeat_start <= 1 and os.path.exists(metric_path_for_reset):
                os.remove(metric_path_for_reset)
        for f, (idx_train, idx_test) in enumerate(idx_train_test):
            repeat_number = f + 1
            if split_mode == 'stratified-holdout' and (repeat_number < repeat_start or repeat_number > repeat_end):
                continue
            print('Fold ' + str(repeat_number) + '/' + str(len(idx_train_test)))
            if split_mode == 'stratified-holdout':
                print('Repeat ' + str(repeat_number) + '/' + str(len(idx_train_test)))
            # idx_train = [i for i in idx_train if (labels[i, c] != -1) & (samples[i] in feats)]
            # idx_test = [i for i in idx_test if (labels[i, c] != -1) & (samples[i] in feats)]
            # idx_train = idx_train[np.where(labels[idx_train,c] != -1)[0]]
            # idx_test = idx_test[np.where(labels[idx_test,c]!=-1)[0]]
            idx_train = np.array(idx_valid)[idx_train]
            idx_test = np.array(idx_valid)[idx_test]
            X_train = [feats[samples[i]] for i in idx_train]
            y_train = labels[idx_train, c]
            X_test = [feats[samples[i]] for i in idx_test]
            y_test = labels[idx_test, c]

            # sanity check
            print(f'Train sample size: {len(y_train)}. Test sample size: {len(y_test)}')
            train_pos, train_neg, train_pos_rate = _binary_count(y_train)
            test_pos, test_neg, test_pos_rate = _binary_count(y_test)
            print('Train negative / positive')
            print(np.array([train_neg, train_pos]))
            print('Test negative / positive')
            print(np.array([test_neg, test_pos]))
            print('Train positive rate: {:.6f}. Test positive rate: {:.6f}'.format(train_pos_rate, test_pos_rate))

            # load trained classifier
            if mi_type is None:
                model_path = out_dir + '_' + 'mean' + '_' + classifier + '_' + cat_name + '_i' \
                             + str(instance_size) + '-' + str(instance_stride) + '_q' + str(quantiles) \
                             + '_fold' + str(f)
            else:
                model_path = out_dir + '_' + mi_type + '_' + classifier + '_' + cat_name + '_i' \
                             + str(instance_size) + '-' + str(instance_stride) + '_q' + str(quantiles) \
                             + '_fold' + str(f)

            if split_mode == 'stratified-holdout':
                split_name = _split_tag(train_size, test_size)
                model_path = os.path.join(repeat_output_dir, 'models', cat_name, split_name, 'repeat_{:03d}.joblib'.format(repeat_number))

            if sample_weight is not None:
                # figure out sample weights
                print('Weighting by ' + sample_weight)
                # discard samples missing a label for sample_weight category
                idx_train = idx_train[np.where(labels_sw[idx_train] != -1)[0]]
                X_train = [feats[samples[i]] for i in idx_train if samples[i] in feats]

                y_train = labels[idx_train, c]
                y_sw = y_train + len(label_names[c]) * labels_sw[idx_train]

                uniq = np.unique(y_sw).tolist()
                counts = np.array([(y_sw == l).sum() for l in uniq])
                counts = counts.sum().astype(float) / (counts * len(counts))
                sw = np.array([counts[uniq.index(y)] for y in y_sw])

                # if load_train and os.path.exists(model_path):
                #     model = load(model_path)
                if mi_type is None:
                    model = LinearClassifier(n_jobs=n_jobs, **options)
                    model.fit(X_train, y_train, calibrate=calibrate, param_search=True, sample_weight=sw)
                elif mi_type in ['median', 'max']:
                    model = SIL(n_jobs=n_jobs, **options)
                    model.fit(X_train, y_train, calibrate=calibrate, param_search=True, sample_weight=sw)
                elif 'quantile' in mi_type:
                    if quantiles is not None:
                        options['quantiles'] = int(quantiles)
                    model = SIL(n_jobs=n_jobs, **options)
                    model.fit(X_train, y_train, calibrate=calibrate, param_search=True, sample_weight=sw)

            else:
                # if load_train and os.path.exists(model_path):
                #     model = load(model_path)
                if mi_type is None:
                    model = LinearClassifier(n_jobs=n_jobs, **options)
                    model.fit(X_train, y_train, calibrate=calibrate, param_search=True)
                elif mi_type in ['median', 'max']:
                    model = SIL(n_jobs=n_jobs, **options)
                    model.fit(X_train, y_train, calibrate=calibrate, param_search=True)
                elif 'quantile' in mi_type:
                    if quantiles is not None:
                        options['quantiles'] = int(quantiles)
                    model = SIL(n_jobs=n_jobs, **options)
                    model.fit(X_train, y_train, calibrate=calibrate, param_search=True)

            p_predict = model.predict(X_test)
            y_predict = np.argmax(p_predict, axis=1)
            acc = sklearn.metrics.accuracy_score(y_test, y_predict)
            if len(y_test) == 1:
                auc = 0.0
            elif len(np.unique(y_train)) == 2:
                auc = sklearn.metrics.roc_auc_score(y_test, p_predict[:, 1])
            else:
                auc = None
            if c_index:
                c_index_continuous = sksurv.metrics.concordance_index_censored(
                    [True] * len(y_test), y_test, -p_predict[:, 1]
                )[0]
                c_index_binary = sksurv.metrics.concordance_index_censored(
                    [True] * len(y_test), y_test, -y_predict
                )[0]
            kappa = sklearn.metrics.cohen_kappa_score(y_test, y_predict)
            classes = np.unique(y_train)
            np.sort(classes)
            confusion = sklearn.metrics.confusion_matrix(y_test, y_predict, labels=classes)
            res.add('acc', acc)
            # res.add('auc', auc)
            # res.add('kappa', kappa)
            sensitivity = float('nan')
            specificity = float('nan')
            if len(label_names[c]) == 2:
                sensitivity = float(np.logical_and(y_test == 1, y_predict == y_test).sum()) / (y_test == 1).sum()
                specificity = float(np.logical_and(y_test != 1, y_predict == y_test).sum()) / (y_test != 1).sum()
                res.add('sensitivity', sensitivity)
                res.add('specificity', specificity)
            res.add('auc', auc if auc is not None else float('nan'))
            if c_index:
                res.add('c_index_continuous', c_index_continuous)
                res.add('c_index_binary', c_index_binary)
            res.add('kappa', kappa)
            res.add('confusion', confusion)

            if len(label_names[c]) > 2:
                classes = np.unique(y_test)
                for cls in classes:
                    cls_name = label_names[c][cls] if cls < len(label_names[c]) else f"class_{cls}"
                    if cls_name not in res.per_class:
                        res.per_class[cls_name] = {}
                    stats_dict = res.per_class[cls_name]

                    cls_acc = sklearn.metrics.accuracy_score(
                        (y_test == cls).astype(int), (y_predict == cls).astype(int)
                    )
                    stats_dict.setdefault('accuracy', []).append(cls_acc)

                    tp = float(np.logical_and(y_test == cls, y_predict == cls).sum())
                    fn = float(np.logical_and(y_test == cls, y_predict != cls).sum())
                    fp = float(np.logical_and(y_test != cls, y_predict == cls).sum())
                    tn = float(np.logical_and(y_test != cls, y_predict != cls).sum())
                    if tp + fn > 0:
                        stats_dict.setdefault('sensitivity', []).append(tp / (tp + fn))
                    if tn + fp > 0:
                        stats_dict.setdefault('specificity', []).append(tn / (tn + fp))

                    if p_predict.shape[1] > cls:
                        try:
                            auc_cls = sklearn.metrics.roc_auc_score(y_test == cls, p_predict[:, cls])
                        except ValueError:
                            auc_cls = float('nan')
                        stats_dict.setdefault('auc', []).append(auc_cls)

            print('accuracy %f auc %f' % (acc, auc))
            print('sensitivity %f specificity %f kappa %f' % (sensitivity, specificity, kappa))
            print(confusion)

            if split_mode == 'stratified-holdout':
                split_name = _split_tag(train_size, test_size)
                split_path = os.path.join(repeat_output_dir, 'splits', cat_name, split_name, 'repeat_{:03d}.tsv'.format(repeat_number))
                pred_path = os.path.join(repeat_output_dir, 'predictions', cat_name, split_name, 'repeat_{:03d}.tsv'.format(repeat_number))
                metric_path = os.path.join(repeat_output_dir, 'metrics', cat_name, split_name + '_repeat_metrics.tsv')
                split_rows = []
                for split_label, split_idx, split_y in [('train', idx_train, y_train), ('test', idx_test, y_test)]:
                    for sample_idx, label_value in zip(split_idx, split_y):
                        split_rows.append({
                            'repeat': repeat_number,
                            'sample_id': samples[sample_idx],
                            'original_index': int(sample_idx),
                            'split': split_label,
                            'label': int(label_value),
                            'label_name': _label_name(label_names[c], label_value),
                        })
                _write_tsv(split_path, ['repeat', 'sample_id', 'original_index', 'split', 'label', 'label_name'], split_rows)

                pred_fields = ['repeat', 'sample_id', 'original_index', 'true_label', 'true_label_name', 'predicted_label', 'predicted_label_name']
                pred_fields += ['prob_{}'.format(i) for i in range(p_predict.shape[1])]
                pred_rows = []
                for row_i, sample_idx in enumerate(idx_test):
                    row = {
                        'repeat': repeat_number,
                        'sample_id': samples[sample_idx],
                        'original_index': int(sample_idx),
                        'true_label': int(y_test[row_i]),
                        'true_label_name': _label_name(label_names[c], y_test[row_i]),
                        'predicted_label': int(y_predict[row_i]),
                        'predicted_label_name': _label_name(label_names[c], y_predict[row_i]),
                    }
                    for prob_i in range(p_predict.shape[1]):
                        row['prob_{}'.format(prob_i)] = float(p_predict[row_i, prob_i])
                    pred_rows.append(row)
                _write_tsv(pred_path, pred_fields, pred_rows)

                if confusion.shape == (2, 2):
                    tn, fp, fn, tp = [int(x) for x in confusion.ravel()]
                else:
                    tn = fp = fn = tp = ''
                metric_row = {
                    'repeat': repeat_number, 'gene': cat_name, 'split': split_name,
                    'train_size_requested': train_size, 'test_size_requested': test_size,
                    'n_train': len(y_train), 'n_test': len(y_test),
                    'train_positive': train_pos, 'train_negative': train_neg, 'train_positive_rate': train_pos_rate,
                    'test_positive': test_pos, 'test_negative': test_neg, 'test_positive_rate': test_pos_rate,
                    'accuracy': acc, 'sensitivity': sensitivity, 'specificity': specificity,
                    'auc': auc if auc is not None else float('nan'), 'kappa': kappa,
                    'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                    'model_file': model_path, 'split_file': split_path, 'prediction_file': pred_path,
                }
                metric_fields = ['repeat', 'gene', 'split', 'train_size_requested', 'test_size_requested', 'n_train', 'n_test',
                                 'train_positive', 'train_negative', 'train_positive_rate', 'test_positive', 'test_negative',
                                 'test_positive_rate', 'accuracy', 'sensitivity', 'specificity', 'auc', 'kappa',
                                 'tn', 'fp', 'fn', 'tp', 'model_file', 'split_file', 'prediction_file']
                _append_tsv(metric_path, metric_fields, metric_row)

            if save_train or split_mode == 'stratified-holdout':
                _mkdir(os.path.dirname(model_path))
                dump(model, model_path)

            if save_first:
                # model_path = out_dir + '_' + classifier + '_' + cat_name
                first_model_path = out_dir + '_' + mi_type + '_' + classifier + '_' + cat_name + '_i' + str(
                    instance_size) + '-' + str(instance_stride) + '_q' + str(quantiles) + '_first'
                data_path = out_dir + '_data_' + cat_name + '_random_state_' + str(random_state)

                dump(model, first_model_path)
                if save_idx:
                    dump([(np.array(samples)[idx_train], X_train, y_train),
                          (np.array(samples)[idx_test], X_test, y_test)],
                         data_path)
                else:
                    dump([(X_train, y_train), (X_test, y_test)], data_path)
                break

            if group is not None:
                # within group class metrics
                l_group = labels_group[idx_test]
                uniq = np.unique(l_group)
                uniq.sort()
                for u in uniq:
                    if u == -1:
                        continue
                    idx = (l_group == u)

                    group_name = '(%s=%s)' % (group, label_names_group[u])
                    res.add('accuracy ' + group_name, sklearn.metrics.accuracy_score(y_test[idx], y_predict[idx]))
                    if len(label_names[c]) == 2:
                        res.add('sensitivity ' + group_name,
                                float(np.logical_and(y_test[idx] == 1, y_predict[idx] == y_test[idx]).sum()) / (
                                        y_test[idx] == 1).sum())
                        res.add('specificity ' + group_name,
                                float(np.logical_and(y_test[idx] != 1, y_predict[idx] == y_test[idx]).sum()) / (
                                        y_test[idx] != 1).sum())
                    if len(np.unique(y_train)) == 2:
                        if (y_test[idx] == 0).sum() == 0 or (y_test[idx] == 1).sum() == 0:
                            auc = 0
                        else:
                            auc = sklearn.metrics.roc_auc_score(y_test[idx], p_predict[idx, 1])
                    else:
                        auc = 0.0
                        for i in range(p_predict.shape[1]):
                            auc += sklearn.metrics.roc_auc_score(y_test[idx] == i, p_predict[idx, i])
                        auc /= p_predict.shape[1]
                    res.add('auc ' + group_name, auc)
                    res.add('kappa ' + group_name, sklearn.metrics.cohen_kappa_score(y_test[idx], y_predict[idx]))
                    # if len(label_names[c]) == 2:
                    #     res.add('sensitivity '+group_name,float( np.logical_and(y_test[idx]==1, y_predict[idx]==y_test[idx]).sum() ) / (y_test[idx]==1).sum() )
                    #     res.add('specificity '+group_name,float( np.logical_and(y_test[idx]!=1, y_predict[idx]==y_test[idx]).sum() ) / (y_test[idx]!=1).sum() )

        print(f'Instance size-stride: {instance_size}-{instance_stride}')
        print(f'Quantiles: {quantiles}')
        print('Cross-validation results')
        res.print_summary()
