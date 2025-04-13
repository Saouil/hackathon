import pandas as pd

# 加载已标注的数据（如果你还没加载）
df = pd.read_csv("C:/Users/Andrew/Hackaton/labeled_prcp_output.csv")

# 如果你之前是在 dataframe 中添加了 prcp_label_from_function 那就直接用
# 否则先从 JSON 再做一遍（这段略去）

# 计算准确率
correct_matches = (df["fuzzy_prcp_label"].str.lower() == df["prcp_label_from_function"].str.lower()).sum()
total = len(df)
accuracy = correct_matches / total

print(f"Total samples: {total}")
print(f"Correct matches: {correct_matches}")
print(f"Accuracy: {accuracy * 100:.2f}%")
