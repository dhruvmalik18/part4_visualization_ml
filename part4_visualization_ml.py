import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("students.csv")

print(df.head())
print(df.shape)
print(df.describe())
print(df['passed'].value_counts())

subjects = ['math','science','english','history','pe']

df['avg_score'] = df[subjects].mean(axis=1)

# ----- Matplotlib -----

df[subjects].mean().plot(kind='bar')
plt.savefig("plot1_bar.png")
plt.close()

plt.hist(df['math'], bins=5)
plt.savefig("plot2_hist.png")
plt.close()

pass_df = df[df['passed']==1]
fail_df = df[df['passed']==0]

plt.scatter(pass_df['study_hours_per_day'], pass_df['avg_score'])
plt.scatter(fail_df['study_hours_per_day'], fail_df['avg_score'])
plt.savefig("plot3_scatter.png")
plt.close()

plt.boxplot([pass_df['attendance_pct'], fail_df['attendance_pct']])
plt.savefig("plot4_box.png")
plt.close()

plt.plot(df['name'], df['math'])
plt.savefig("plot5_line.png")
plt.close()

# ----- Seaborn -----

sns.barplot(x='passed', y='math', data=df)
plt.savefig("plot6_seaborn_bar.png")
plt.close()

sns.scatterplot(x='attendance_pct', y='avg_score', hue='passed', data=df)
plt.savefig("plot7_seaborn_scatter.png")
plt.close()

# ----- Machine Learning -----

features = ['math','science','english','history','pe','attendance_pct','study_hours_per_day']

X = df[features]
y = df['passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
# Seaborn provides better visualization with less code and built-in styling.
# Matplotlib gives more control but requires more customization.
# Seaborn is easier for quick analysis, while Matplotlib is more flexible.