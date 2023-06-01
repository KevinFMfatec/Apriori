from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

data = pd.read_csv('groceries - groceries.csv')

# Criar uma lista de transações
transactions = []
for index, row in data.iterrows():
    transaction = []
    for i in range(1, 33):  # Assume que existem 32 colunas de itens (Item 1 até Item 32)
        item = row[f'Item {i}']
        if pd.notna(item):
            transaction.append(item)
    transactions.append(transaction)

# Criar DataFrame com as transações
grouped_df = pd.DataFrame({'Transaction': transactions})

# Codificar as transações usando one-hot encoding
encoded_df = grouped_df['Transaction'].str.join('|').str.get_dummies()

# Aplicar o algoritmo Apriori
frequent_items = apriori(encoded_df, min_support=0.01, use_colnames=True)

# Gerar as regras de associação
rules = association_rules(frequent_items, metric='confidence', min_threshold=0.5)

# Imprimir o resultado
print(rules)