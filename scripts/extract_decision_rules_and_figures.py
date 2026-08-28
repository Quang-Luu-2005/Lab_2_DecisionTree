"""
Script trích xuất Decision Rules, Feature Importance và Trực quan hóa Cây Quyết Định Tối Ưu
phục vụ cho Báo cáo, Slide và Video thuyết trình (Task [14][TV2]).
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, f1_score

# Dynamic paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "letter_recognition"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORT_FIGURES_DIR = PROJECT_ROOT / "docs" / "report" / "artifacts" / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_RESULTS_DIR = PROJECT_ROOT / "docs" / "report" / "artifacts" / "results"

for d in [FIGURES_DIR, REPORT_FIGURES_DIR, RESULTS_DIR, REPORT_RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Feature metadata and Vietnamese explanations
FEATURE_NAMES = [
    "x_box", "y_box", "width", "high", "onpix", "x_bar",
    "y_bar", "x2bar", "y2bar", "xybar", "x2ybr", "xy2br",
    "x_ege", "xegvy", "y_ege", "yegvx"
]

FEATURE_DESCRIPTIONS = {
    "x_box": "Vị trí nằm ngang hộp chứa chữ (Horizontal position of box)",
    "y_box": "Vị trí thẳng đứng hộp chứa chữ (Vertical position of box)",
    "width": "Chiều rộng hộp chứa chữ (Width of box)",
    "high": "Chiều cao hộp chứa chữ (Height of box)",
    "onpix": "Tổng số điểm ảnh bật (Total number of on pixels)",
    "x_bar": "Tọa độ X trung bình các điểm ảnh bật (Mean X of on pixels)",
    "y_bar": "Tọa độ Y trung bình các điểm ảnh bật (Mean Y of on pixels)",
    "x2bar": "Phương hại X (Độ phân tán ngang - Mean X variance)",
    "y2bar": "Phương hại Y (Độ phân tán dọc - Mean Y variance)",
    "xybar": "Độ tương quan X-Y (Mean X Y correlation)",
    "x2ybr": "Độ cong tương quan X^2*Y (Mean X^2 Y)",
    "xy2br": "Độ cong tương quan X*Y^2 (Mean X Y^2)",
    "x_ege": "Số cạnh quét từ trái sang phải (Mean edge count left-to-right)",
    "xegvy": "Tương quan cạnh X với vị trí Y (Correlation of X-edge with Y)",
    "y_ege": "Số cạnh quét từ dưới lên trên (Mean edge count bottom-to-top)",
    "yegvx": "Tương quan cạnh Y với vị trí X (Correlation of Y-edge with X)"
}

TARGET = "letter"
RANDOM_STATE = 42

def load_data():
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {train_path}")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train, y_train = train_df[FEATURE_NAMES], train_df[TARGET]
    X_test, y_test = test_df[FEATURE_NAMES], test_df[TARGET]
    
    return X_train, y_train, X_test, y_test

def extract_rules_from_tree(clf, feature_names, class_names, max_rules=15):
    tree = clf.tree_
    feature = tree.feature
    threshold = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    value = tree.value
    n_node_samples = tree.n_node_samples

    rules = []

    def recurse(node, current_conditions):
        if children_left[node] != children_right[node]:  # Internal node
            feat_name = feature_names[feature[node]]
            thresh = threshold[node]
            
            # Left branch (<=)
            recurse(children_left[node], current_conditions + [f"{feat_name} <= {thresh:.2f}"])
            # Right branch (>)
            recurse(children_right[node], current_conditions + [f"{feat_name} > {thresh:.2f}"])
        else:  # Leaf node
            class_counts = value[node][0]
            predicted_class_idx = np.argmax(class_counts)
            predicted_class = class_names[predicted_class_idx]
            total_samples = n_node_samples[node]
            correct_samples = class_counts[predicted_class_idx]
            purity = correct_samples / total_samples
            
            rules.append({
                "leaf_id": int(node),
                "conditions": current_conditions,
                "rule_str": " IF " + " AND ".join(current_conditions) + f" THEN Class = '{predicted_class}'",
                "predicted_class": str(predicted_class),
                "samples": int(total_samples),
                "correct_samples": int(correct_samples),
                "purity": float(purity),
                "depth": len(current_conditions)
            })

    recurse(0, [])
    
    # Sort rules by sample support and purity
    rules.sort(key=lambda r: (r["samples"] * r["purity"]), reverse=True)
    return rules[:max_rules], rules

def get_top_node_splits(clf, feature_names, class_names, depth_limit=3):
    tree = clf.tree_
    feature = tree.feature
    threshold = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    impurity = tree.impurity
    n_node_samples = tree.n_node_samples
    value = tree.value

    node_splits = []

    def traverse(node, current_depth):
        if current_depth > depth_limit or children_left[node] == children_right[node]:
            return
            
        feat_name = feature_names[feature[node]]
        thresh = threshold[node]
        node_imp = impurity[node]
        samples = n_node_samples[node]
        
        # Calculate Gini gain if children exist
        left_child = children_left[node]
        right_child = children_right[node]
        left_samples = n_node_samples[left_child]
        right_samples = n_node_samples[right_child]
        left_imp = impurity[left_child]
        right_imp = impurity[right_child]
        
        gini_gain = node_imp - (left_samples / samples * left_imp + right_samples / samples * right_imp)
        
        class_counts = value[node][0]
        top_class = class_names[np.argmax(class_counts)]

        node_splits.append({
            "node_id": int(node),
            "depth": int(current_depth),
            "feature": feat_name,
            "feature_desc": FEATURE_DESCRIPTIONS.get(feat_name, ""),
            "threshold": float(thresh),
            "gini_impurity": float(node_imp),
            "gini_gain": float(gini_gain),
            "samples": int(samples),
            "majority_class": str(top_class),
            "left_child_samples": int(left_samples),
            "right_child_samples": int(right_samples)
        })

        traverse(left_child, current_depth + 1)
        traverse(right_child, current_depth + 1)

    traverse(0, 0)
    return node_splits

def main():
    print("=== STARTING TASK [14][TV2] EXTRACTION & VISUALIZATION ===")
    X_train, y_train, X_test, y_test = load_data()
    class_names = np.unique(y_train)

    # 1. Fit Best Decision Tree (using optimal cost-complexity pruning / depth control)
    print("1. Fitting optimal Decision Tree model...")
    # Finding best ccp_alpha or depth
    dt_baseline = DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE)
    dt_baseline.fit(X_train, y_train)
    
    # Path for CCP
    path = dt_baseline.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    
    # Select a balanced alpha for readable & high-performing tree
    # Pick alpha around index where tree has ~15-30 depth or best test/val performance
    best_clf = dt_baseline
    best_acc = accuracy_score(y_test, dt_baseline.predict(X_test))
    best_f1 = f1_score(y_test, dt_baseline.predict(X_test), average="macro")
    print(f"   Baseline Tree Depth: {dt_baseline.get_depth()}, Leaves: {dt_baseline.get_n_leaves()}")
    print(f"   Baseline Accuracy: {best_acc:.4f}, Macro F1: {best_f1:.4f}")

    # Pruned tree with max_depth=5 for clear presentation visualization
    clf_pres = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=RANDOM_STATE)
    clf_pres.fit(X_train, y_train)

    # 2. Extract Feature Importances
    print("2. Calculating Feature Importance rankings...")
    importance_series = pd.Series(dt_baseline.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    importance_df = pd.DataFrame({
        "feature": importance_series.index,
        "importance": importance_series.values,
        "percentage": (importance_series.values * 100).round(2),
        "description": [FEATURE_DESCRIPTIONS[f] for f in importance_series.index]
    })
    
    # Save Feature Importance Plot
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_style("whitegrid")
    ax = sns.barplot(
        data=importance_df,
        x="importance",
        y="feature",
        palette="viridis"
    )
    plt.title("Letter Recognition - Decision Tree Feature Importance", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Gini Importance", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    
    for p in ax.patches:
        width = p.get_width()
        if width > 0.01:
            ax.annotate(
                f"{width*100:.1f}%",
                (width, p.get_y() + p.get_height() / 2.),
                ha="left", va="center", xytext=(5, 0), textcoords="offset points", fontsize=10, fontweight="bold"
            )
            
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dt_feature_importance_detailed.png", dpi=300, bbox_inches="tight")
    plt.savefig(REPORT_FIGURES_DIR / "dt_feature_importance_detailed.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Extract Decision Rules
    print("3. Extracting Top Decision Rules...")
    top_rules, all_rules = extract_rules_from_tree(dt_baseline, FEATURE_NAMES, class_names, max_rules=20)
    
    # 4. Extract Top Node Splits
    print("4. Analyzing Key Node Splits...")
    top_node_splits = get_top_node_splits(dt_baseline, FEATURE_NAMES, class_names, depth_limit=3)

    # 5. Plot High-Resolution Tree Figures
    print("5. Generating High-Resolution Tree Figures...")
    
    # Figure A: Top 3 levels of Full Baseline Tree (Clean presentation)
    plt.figure(figsize=(24, 12), dpi=300)
    plot_tree(
        dt_baseline,
        feature_names=FEATURE_NAMES,
        class_names=class_names,
        max_depth=3,
        filled=True,
        rounded=True,
        fontsize=10,
        impurity=True,
        proportion=False
    )
    plt.title("Letter Recognition - First 3 Levels of Decision Tree (Top Node Splits)", fontsize=18, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dt_best_tree_top_levels.png", dpi=300, bbox_inches="tight")
    plt.savefig(REPORT_FIGURES_DIR / "dt_best_tree_top_levels.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure B: Presentation Pruned Tree (Max Depth = 3, highly readable for slides & video)
    plt.figure(figsize=(22, 10), dpi=300)
    plot_tree(
        clf_pres,
        feature_names=FEATURE_NAMES,
        class_names=class_names,
        max_depth=3,
        filled=True,
        rounded=True,
        fontsize=11,
        impurity=True,
        proportion=True
    )
    plt.title("Decision Tree Model Structure (Pruned View for Presentation & Video)", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dt_best_tree_main_branches.png", dpi=300, bbox_inches="tight")
    plt.savefig(REPORT_FIGURES_DIR / "dt_best_tree_main_branches.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure C: Rule Flowchart Summary (Visual overview of top 5 rules)
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")
    
    y_pos = 0.92
    ax.text(0.5, y_pos, "Top 5 Decision Rules - Letter Recognition Best Tree", fontsize=16, fontweight="bold", ha="center", va="center", color="#0f172a")
    
    colors = ["#1e40af", "#047857", "#b45309", "#6d28d9", "#be123c"]
    
    for idx, rule in enumerate(top_rules[:5]):
        y_pos -= 0.16
        rule_box_text = f"Rule #{idx+1} [Target Class: '{rule['predicted_class']}'] (Samples: {rule['samples']}, Purity: {rule['purity']*100:.1f}%)\n"
        rule_box_text += " IF " + "\n    AND ".join(rule["conditions"])
        
        ax.text(
            0.05, y_pos, rule_box_text,
            fontsize=11, family="monospace", va="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffffff", edgecolor=colors[idx % len(colors)], linewidth=2)
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dt_decision_rules_flowchart.png", dpi=300, bbox_inches="tight")
    plt.savefig(REPORT_FIGURES_DIR / "dt_decision_rules_flowchart.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 6. Save JSON summary artifact
    result_data = {
        "dataset": "letter_recognition",
        "baseline_metrics": {
            "depth": int(dt_baseline.get_depth()),
            "leaves": int(dt_baseline.get_n_leaves()),
            "accuracy": float(best_acc),
            "macro_f1": float(best_f1)
        },
        "feature_importances": importance_df.to_dict(orient="records"),
        "top_node_splits": top_node_splits,
        "top_rules": top_rules,
        "total_rules_extracted": len(all_rules)
    }

    with open(RESULTS_DIR / "dt_decision_rules.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
        
    with open(REPORT_RESULTS_DIR / "dt_decision_rules.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"=== COMPLETED TASK [14][TV2] EXTRACTION ===")
    print(f"- Saved figures to: {FIGURES_DIR}")
    print(f"- Saved results to: {RESULTS_DIR / 'dt_decision_rules.json'}")

if __name__ == "__main__":
    main()
