import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("cleaned_t20_dataset.csv")
df.columns = df.columns.str.strip()

# Drop rows with missing target
df = df.dropna(subset=['home_score'])

# Convert categorical variables
df['toss_decision'] = df['decision'].astype('category').cat.codes
df['venue'] = df['venue_name'].astype('category').cat.codes
df['match_days'] = pd.to_numeric(df['match_days'], errors='coerce').fillna(0)

# Features and target
features = ['toss_decision', 'venue', 'match_days']
target = 'home_score'

X = df[features]
y = df[target]

# Add intercept
X = sm.add_constant(X)

# Fit model
model = sm.OLS(y, X).fit()

# Print summary
print(model.summary())

# Plot residuals
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.grid(True)
plt.show()
