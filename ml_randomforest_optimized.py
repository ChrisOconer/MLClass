import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from build_ml_dataset import build_ml_dataset
from feature_engineering import extract_advanced_features, select_best_features, get_feature_names

def plot_confusion(y_true, y_pred, classes, title='Confusion matrix'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(np.arange(len(classes)), classes)
    plt.yticks(np.arange(len(classes)), classes)
    plt.colorbar(im)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(importances, feature_names, topn=20, title='Feature Importance'):
    idx = np.argsort(importances)[::-1][:topn]
    plt.figure(figsize=(8,6))
    plt.barh(np.array(feature_names)[idx][::-1], importances[idx][::-1])
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

def analyze_misclassified(X_val, Y_val, y_pred, original_X_val=None):
    """输出误判样本编号和标签"""
    mis_idx = np.where(Y_val != y_pred)[0]
    print(f"共有 {len(mis_idx)} 个误判样本：")
    for i in mis_idx:
        print(f"Index: {i}, True: {Y_val[i]}, Pred: {y_pred[i]}")
        # 可选：展示原始时间序列的人体中心轨迹、某些关节轨迹等
        if original_X_val is not None:
            frames = original_X_val[i]
            center = np.mean(frames.reshape(frames.shape[0], -1, 2), axis=1)
            plt.plot(center[:,0], center[:,1])
            plt.title(f"Sample {i} Center Trajectory (True:{Y_val[i]}, Pred:{y_pred[i]})")
            plt.show()

if __name__ == "__main__":
    # 1. 读取数据
    X_raw_train, Y_train = build_ml_dataset('normalized_new_17_2d.h5', group='train')
    X_raw_val, Y_val = build_ml_dataset('normalized_new_17_2d.h5', group='validation')
    # 2. 高级特征工程
    X_train_feat = extract_advanced_features(X_raw_train, n_segments=3)
    X_val_feat = extract_advanced_features(X_raw_val, n_segments=3)
    # 3. 特征选择
    X_train_sel, X_val_sel, selector = select_best_features(X_train_feat, Y_train, X_val_feat, k=300)
    feature_names = get_feature_names(X_train_feat.shape)
    selected_feature_names = np.array(feature_names)[selector.get_support()]
    # 4. 多模型套餐
    models = {
        'RF': RandomForestClassifier(n_estimators=300, max_depth=10, max_features='sqrt', min_samples_leaf=2, class_weight='balanced', random_state=42),
        'XGB': xgb.XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    }
    # 5. 自动调参（以RF为例，可扩展到XGB/SVM）
    param_grid_rf = {'n_estimators':[200,300, 500],'max_depth':[8,10,12, 15]}
    gs_rf = GridSearchCV(models['RF'], param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1)
    gs_rf.fit(X_train_sel, Y_train)
    best_rf = gs_rf.best_estimator_
    print("Best RF Params:", gs_rf.best_params_)
    # XGBoost调参
    param_grid_xgb = {'n_estimators':[200,300, 500],'max_depth':[8,10, 15],'learning_rate':[0.05,0.1, 0.2]}
    gs_xgb = GridSearchCV(models['XGB'], param_grid_xgb, cv=3, scoring='accuracy', n_jobs=-1)
    gs_xgb.fit(X_train_sel, Y_train)
    best_xgb = gs_xgb.best_estimator_
    print("Best XGB Params:", gs_xgb.best_params_)
    # SVM调参
    param_grid_svm = {'C':[0.1,1,10, 100],'gamma':['scale',0.01,0.1, 1]}
    gs_svm = GridSearchCV(models['SVM'], param_grid_svm, cv=3, scoring='accuracy', n_jobs=-1)
    gs_svm.fit(X_train_sel, Y_train)
    best_svm = gs_svm.best_estimator_
    print("Best SVM Params:", gs_svm.best_params_)
    # 6. 融合模型（Voting）
    voting = VotingClassifier(estimators=[
        ('rf', best_rf), ('xgb', best_xgb), ('svm', best_svm)
    ], voting='soft', weights=[2,3,1])
    voting.fit(X_train_sel, Y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # 7. 按模型分别输出CV和验证集表现
    for name, clf in [('RF', best_rf), ('XGB', best_xgb), ('SVM', best_svm), ('Voting', voting)]:
        cv_scores = cross_val_score(clf, X_train_sel, Y_train, cv=cv, scoring='accuracy')
        y_pred = clf.predict(X_val_sel)
        acc = accuracy_score(Y_val, y_pred)
        print(f"\n>>> {name} <<<")
        print(f"CV Scores: {cv_scores}, Mean: {cv_scores.mean():.4f}")
        print("Validation Accuracy:", acc)
        print(classification_report(Y_val, y_pred))
        plot_confusion(Y_val, y_pred, [str(i) for i in np.unique(Y_train)], title=f'{name} Confusion Matrix')
        # 特征重要性（仅RF/XGB支持）
        if hasattr(clf, 'feature_importances_'):
            plot_feature_importance(clf.feature_importances_, selected_feature_names, topn=20, title=f'{name} Feature Importance')
        # 误判分析
        analyze_misclassified(X_val_sel, Y_val, y_pred, original_X_val=X_raw_val)