import requests
import ast

url = "https://investors-exchange-iex-trading.p.rapidapi.com/stock/crm/time-series"
url = "https://investors-exchange-iex-trading.p.rapidapi.com/stock/aapl/chart"

headers = {
    'x-rapidapi-key': "e65d9a9d76msh12c87111bea3072p1549f2jsne5060e9afd9d",
    'x-rapidapi-host': "investors-exchange-iex-trading.p.rapidapi.com"
    }

response = requests.request("GET", url, headers=headers)

print(response.text)

x = response.text
y = ast.literal_eval(x)



url = "https://investors-exchange-iex-trading.p.rapidapi.com/stock/aapl/chart/ytd"


url_base='https://cloud.iexapis.com/stable/stock/'
ticker = 'FB'
url_chart = '/chart/date/'
date = '20191107'
token = '?token=pk_98ce4b891f0b439d99f2a93a84aa125a'

query = url_base+ticker+url_chart+date+token
response = requests.request("GET", query, headers=headers)
x = response.text
y = ast.literal_eval(x)