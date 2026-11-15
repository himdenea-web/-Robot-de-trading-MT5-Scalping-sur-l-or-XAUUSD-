import MetaTrader5 as mt5

print("Connexion :", mt5.initialize())
print(mt5.account_info())
mt5.shutdown()