import re

with open('scripts/step_04_exploratory_data_analysis.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix garbled unicode in the plot title
content = re.sub(
    r'plt\.title\("Multi-Asset Stock Price Trends \(2010.*?2024\)"\)', 
    'plt.title("Multi-Asset Stock Price Trends (2010-2024)")', 
    content
)

with open('scripts/step_04_exploratory_data_analysis.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed garbled Unicode in step_04!')
