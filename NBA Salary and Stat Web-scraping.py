#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
from bs4 import BeautifulSoup as bs


# In[7]:


url = "https://hoopshype.com/salaries/players/2020-2021/"


# In[8]:


r = requests.get(url)


# In[9]:


soup = bs(r.content)


# In[10]:


soup


# In[41]:


body = soup.find_all("tr")[1]
body


# In[42]:


name = body.find_all(class_="name")[0].get_text().strip()
name


# In[33]:


i = 1
player_name = []
while i <= 578:
    body = soup.find_all("tr")[i]
    name = body.find_all(class_="name")[0].get_text().strip()
    player_name.append(name)
    i += 1
player_name


# In[ ]:





# In[48]:


salary = body.find_all("td")[3].get_text().strip()
salary


# In[49]:


i = 1
player_salary = []
while i <= 578:
    body = soup.find_all("tr")[i]
    salary = body.find_all("td")[3].get_text().strip()
    player_salary.append(salary)
    i += 1
player_salary


# In[ ]:





# In[122]:


url3 = "https://www.basketball-reference.com/leagues/NBA_2021_totals.html"


# In[123]:


r3 = requests.get(url3)


# In[124]:


soup3 = bs(r3.content)


# In[125]:


print(soup3.prettify())


# In[129]:


body3 = soup3.find_all("tbody")[0]
print(body3.prettify())


# In[139]:


stat_class = body3.find_all(class_="full_table")[539]
stat_class


# In[140]:


stat_class.get_text(" ,")


# In[141]:


stat = []

table_rows = body3.find_all(class_="full_table")
for tr in table_rows:
    td = tr.find_all('td')
    row = [str(tr.string) for tr in td]
    stat.append(row)
print(stat)


# In[216]:


pd.options.display.max_rows = 999
pd.options.display.max_columns = 999



# In[200]:


columns = ["Player", "Pos", "Age", "Team", "G", "GS", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]


# In[217]:


df_stat = pd.DataFrame(stat, columns=columns)
df_stat = df_stat.set_index("Player")
df_stat


# In[ ]:





# In[50]:


import pandas as pd


# In[53]:


df = pd.DataFrame(player_name)
df["Salary"] = player_salary


# In[55]:


df.head()


# In[60]:


df = df.rename(columns={0: "Player"})


# In[62]:


df["Year"] = 2021


# In[207]:


df = df.set_index("Player")


# In[208]:


df.head()


# In[94]:


df.info()


# In[ ]:





# In[220]:


df20 = pd.concat([df_stat, df], axis=1)
df20.to_excel("nba_20.xlsx")


# In[226]:


df_20 = pd.read_csv("/Users/kevinjoo/Downloads/NBA20 edit-1.csv")
df_20.info()


# In[ ]:





# In[ ]:





# In[64]:


url2 = "https://hoopshype.com/salaries/players/2019-2020/"


# In[70]:


r2 = requests.get(url2)


# In[71]:


soup2 = bs(r2.content)


# In[82]:


body2 = soup2.find_all("tr")[1]
body2


# In[80]:


x = 1
player_name2 = []
while x <= 513:
    body2 = soup2.find_all("tr")[x]
    name2 = body2.find_all(class_="name")[0].get_text().strip()
    player_name2.append(name2)
    x += 1
player_name2


# In[84]:


salary2 = body2.find_all("td")[2].get_text().strip()
salary2


# In[85]:


i = 1
player_salary2 = []
while i <= 513:
    body2 = soup2.find_all("tr")[i]
    salary2 = body2.find_all("td")[2].get_text().strip()
    player_salary2.append(salary2)
    i += 1
player_salary2


# In[236]:


df2 = pd.DataFrame(player_name2)
df2["Salary"] = player_salary2
df2 = df2.rename(columns={0: "Player"})
df2["Year"] = 2020
df2 = df2.set_index("Player")


# In[237]:


df2.head()


# In[93]:


df2.info()


# In[91]:


df_main = pd.concat([df, df2])


# In[92]:


df_main.info()


# In[ ]:





# In[227]:


url4 = "https://www.basketball-reference.com/leagues/NBA_2020_totals.html"


# In[228]:


r4 = requests.get(url4)


# In[229]:


soup4 = bs(r4.content)


# In[230]:


soup4


# In[231]:


body4 = soup4.find_all("tbody")[0]


# In[232]:


stat_class2 = body4.find_all(class_="full_table")[528]
stat_class2


# In[233]:


stat2 = []

table_rows = body4.find_all(class_="full_table")
for tr in table_rows:
    td = tr.find_all('td')
    row = [str(tr.string) for tr in td]
    stat2.append(row)
print(stat2)


# In[234]:


df_stat2 = pd.DataFrame(stat2, columns=columns)
df_stat2 = df_stat2.set_index("Player")
df_stat2


# In[238]:


df19 = pd.concat([df_stat2, df2], axis=1)
df19.to_excel("nba_19.xlsx")


# In[ ]:





# In[239]:


df_19 = pd.read_csv("/Users/kevinjoo/Downloads/NBA19 edit.csv")
df_19.info()


# In[ ]:





# In[472]:


df_main = df_20.append(df_19)

