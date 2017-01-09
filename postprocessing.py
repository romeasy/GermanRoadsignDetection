import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

date_string = '2017_01_05_17_37_31'
with open('training/' + date_string + '/accuracies.pkl', 'r') as f:
    test_acc, train_acc, test_ce = pickle.load(f)

fig = plt.figure(figsize=(14,8))
plt.plot(test_acc, color='green', label='Accuracy on the test set')
plt.plot(train_acc, color='red', label='Accuracy on the training set')
plt.legend(loc="lower right")
fig.savefig('accuracy_supergood.png', dpi=600)

print "Final Accuracy on Test Set: " + str(test_acc[-1])
print "Final Accuracy on Training Set: " + str(train_acc[-1])

import cnn

# load test set
with open('test_data_gray_norm_aug.pkl', 'rb') as test_handle:
    test_set, test_labels = pickle.load(test_handle)

path_to_model = 'training/' + date_string + '/model.ckpt'
model = cnn.cNN()
print test_set.shape
print test_labels.shape
predictions = model.load_model_and_evaluate('training/' + date_string + '/model.ckpt', test_set)

from sklearn.metrics import roc_curve, auc
from scipy import interp

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for c in range(model.n_classes):
    fpr[c], tpr[c], _ = roc_curve(y_true=test_labels[:, c], y_score=predictions[:, c])
    roc_auc[c] = auc(fpr[c], tpr[c])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_labels.ravel(), predictions.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(model.n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(model.n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= model.n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
fig = plt.figure(figsize=(14, 8))

plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         linewidth=2)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         linewidth=2)
"""
for i in range(model.n_classes):
	if i in [0, 19, 28, 34, 42]:
		plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
			                           ''.format(i, roc_auc[i]))
"""

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of imperfect classes')
plt.legend(loc="lower right")
fig.savefig('avg_roc_supergood.png', dpi=600)
plt.show()

