import re


# 定义SQL关键字的正则表达式
sql_keywords_regex = r'\b(SELECT|SUM|FROM|WHERE|JOIN|ON|GROUP BY|ORDER BY|HAVING|LIMIT)\b'

# 函数：提取SQL语句中的关键字
def extract_keywords(sql):
    return re.findall(sql_keywords_regex, sql, re.IGNORECASE)

# 示例SQL语句
generated_sql = "SELECT * FROM users WHERE age > 20"
correct_sql = "SELECT name, age FROM users WHERE age > 20 ORDER BY age DESC"

# 提取关键字
generated_keywords = extract_keywords(generated_sql)
correct_keywords = extract_keywords(correct_sql)

# 计算匹配概率
match_probability = len(set(generated_keywords) & set(correct_keywords)) / len(set(correct_keywords))

print(f"Generated SQL Keywords: {generated_keywords}")
print(f"Correct SQL Keywords: {correct_keywords}")
print(f"Match Probability: {match_probability:.2f}")
