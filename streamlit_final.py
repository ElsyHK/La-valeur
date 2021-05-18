import pandas as pd
import numpy as np
import streamlit as st
import plotly
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
import openpyxl

# Define a function to load data for faster computation
@st.cache(allow_output_mutation=True)
def load_data(path):
    df = pd.read_excel(path, engine='openpyxl')
    return df

def load_data1(path):
    df = pd.read_csv(path, encoding='utf-8')
    return df

def freq_itms(onehot_df):
    freq_items =apriori(onehot_df, min_support=0.006, use_colnames=True,verbose=1)
    return freq_items

#sidebar design
st.sidebar.subheader("App Designer: Elsy Hobeika")
st.sidebar.write("")

# Side Bar Menu
add_selectbox = st.sidebar.selectbox(
    'MENU',
    ('Overview','Sales','Supplier Analysis','Customer Analysis','Recommendation System')
)

# Load Data
url1 = 'https://drive.google.com/file/d/1D0qj5YR9-KTd11f-D4fKW6bHf1L3n3rS/view?usp=sharing'
path1 = 'https://drive.google.com/uc?export=download&id='+url1.split('/')[-2]

url2 = 'https://drive.google.com/file/d/1y4kOlIxbzOiDMxFw7i8_oCW27tguLtvf/view?usp=sharing'
path2 = 'https://drive.google.com/uc?export=download&id='+url2.split('/')[-2]
df_encoding = load_data1(path2)


#upload file
upload_file = st.sidebar.file_uploader("Upload Data", type=['CSV','xlsx'])

if upload_file is None:
    df_customers2019 = load_data(path1)

if upload_file is not None:
    df_customers2019 = load_data(upload_file)


# Remove Hours from the date
df_customers2019['Date'] = pd.to_datetime(df_customers2019['Date']).dt.date

# Define dataframe Client Divers
Client_divers = df_customers2019[df_customers2019['Customer Description'] == 'CLIENT DIVERS']
# Define dataframe loyal custumers
loyal_customer = df_customers2019[df_customers2019['Customer Description'] != 'CLIENT DIVERS']



# Overview Page
if add_selectbox == 'Overview':
    col1, col2, col3 = st.beta_columns([1,6,1])
    with col1:
        st.write("")
    with col2:
        st.image("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAATcAAACiCAMAAAATIHpEAAAA7VBMVEX///9NTU8LSZoObrdJSUtycnQANpM/P0EANJLMzMwAOZQAPZUAP5bGxsdCQkQAYrIAabUAQpdfX2EAZbNpaWoAYbL2+Ps6OjwARZjw8PA3XqPi6fN8fH22trfg4+1SUlTR1uVIaaeaqMmyvNUjUp55jbrO2uqBlL4UTJtWdK5gebDw8/gALpBxh7e6w9lUisOOnsPFzN8yWqGksM7d3d2rq6uJiYpDZaYmdLm8zeSPrtTW1tabm5x3n82xxuBgkcaHqdGeuNk9fr4AJo4AV658oc5Vf7lplccgdLoxMTMmJiqhoaGwsLGFhYYAG4sU/JrdAAAUqElEQVR4nO1da0ObyrpGsAQBk2AMQaJIEmJiSGIu1ntta1dPXad2//+fs+fCwAwzEFxbXUHzfGgRhmHm4b0PAUnaoKRot5/TOrRfaxylw+w5jb+/1ijKh9m0eNvrD89bGG95VuGTbCVR6jCn3TvGdbLpDoqe9PMs2X6GlL4nfEk2l2O/2Dknf93H2/78ZcdTFnxN9KytXRY750pJThr4LzygkuCMVlS1W+SUm9rX5I/hs8KX94Mvv5LtpWF6q8/w6o1ETb3xB+Xt52MSwXqadbr6jLN6IzllOf6gAfD35knyx6Wjr3SP3xt1Sk076msMqgT43vyW/NE1ZGtFPNZuKo3EJLYt/XWGtfa4bzwkf3i6bE3y25/VlXqimj3dfKVxrTuuGzXKsl86spEbkN03lDrlSUZmwdDl3eGkUadyTaCoziLH0rcVhVZT23HcVxzbOuOkptCKasqykVMY+VVXFMoBz3Vr9IpjW2e0a0qTcgVAUeVxZhB3XVOUOpWZTaxV5vDdAvBGhbFSFwick2WzbIVVU8mSrcK1gHeGsKnUqeqGZ8hAU5fitlBLaTX11Y/Lm9QEqkcpagB4kx1h8nTSUFg1HViyWSijfY+os6o3A4oqC629fYXUlEovAMfGh+UNsEFHZJ4GBU6Ubn2DWqrUqabQ+fZef4TriQdAxhUVsi0cSJzMBXFIS5X6z2QPCPaEBH8MgMRJqVHKhxRVNrn4Amkp07IDGNb9NxjiWgLyVv+W/A1yVKSpPtvsGxI3RaEaWoJmHwcouKBSBhT6ApfKJlAepo0meAlDFrVAofN9AvFWo6aPFTWVbj0gp6DQxboR5Ff9oOXeyE3Sub1vyJg4isvvkZY+JO4iRM20D1ruJeHFFbXHxR7V6cR72ljaGDWdYt7ebJzrhu+IEzq3jxRV1uPY7CzijQ56JxYi9y2H+jYoaHkwb4yiYo8qxzXz+xqmjZFKrM3Bi45lPbCcFXl24xrHs1RuL8mEN5xuhZG0KfWbpM1cF3jdrEt8KRVvkj8crX4K4TqKMCiOB1ZEHM4GvhLeGl66DWUDs9C+oXOMcsCe/FjMVkRYOH9iinC+GvHmBHZMbEpNA9xgxXqrff21qZzkt1lLTHXdOs0Vuog3OrePSIGxyEAKHxWBmkbBSn65t31z1Wj+KueDXmFHt4zhLNvAtCOjT+f2A5MQp/lfiJYqTT44zuPt+qxRrzfK+4DhzHBkczyaZkSohDc6Y489quxcxrQJkrHscm/7RmnUlfpDuRwCCz+AiwZ6ILZ0IdFCynrbUTEJwI1po0MVL/IcpnDty74HogYF9NuLzuPNYU+Q/JjmqaBaRnhjrH6iqAlvDUp2liQXE6xEtL9dIdbqyjV/cK0Rch6sZ+Aih2FxQmc/KAJFNTjemAjv1Mni7f6hgTW7cZZ2CPbaP5p5fZPe03YjJkyt02MsXcwb7S6lWFET3ig1DQmtqXKv960eBS0Ch+D1/9dpvT5uzjgvMDMiKhx9OKCF7kxk9/sWxxulplNdwFt4f1aLM4sHXuQXZQhIfvHR5lwmXABLRwkdSdqVJsXM3EjxVqdZncQJhU92nfxUGrHrrf3kbls/+ymAtcJDk1MU+zQOL6Cl60cTidMoWhOltLwxB2PrF/Fmx1YNNa1xDsFzf6y9ccMIH5tfuZu+tOL4AsRe2mUPqk7Cm0hRCW+0N53H/KNy78mXWpyKCR2CNDW00qyzntTrV5yueq6REAcsndz3pZ+xpNC5PQl9CW9C4yerYXj/UKsrNG28nA80o0SP39yDNIfPcvqqTMMyOv+XTJnK7SWH4Y3paRifHvxSaFED7R5F98pxy1RO/9kAOsMNeO5YDHOOnEyaeuCZCFXEG10PToI7R2FEDTD/i7teT3dkpwyuNAF8Fpf3qyHlHhCG8bQfeUXFvDFBL6mjyw4ra+Bi9+mL2ROtfIus4ZXQ3khd06F5cxaxuFCO0A4o3hgNdsnZQUrYzrgsHtlTPeOxsPWFV0fT4XTHX5g0cbGm0g9pRYrqch7DI+cOrxjaalySIvWg/zZKkCekcY8eWeN11Z4w7iHWVLoIh6MNlzN8JKd3GNsmuIiEygnW6lL6GuIGmaAar6tT2j04ROPo3B4rqsupaZTTJ9qNWvDBooeE2gnK5RMIcFQriEXbp1Qo5wgWlnE2hfWUOpvk9MFjwlq9zjkErKPl8wkENn6yQxADS12NN3F0EQ7lqG5aTXsGp6UNQVk3ihPV0j5X2CaT43XVHybuIdJUOreXCG+Mmo6cNG21b1zPXuR49BI/Nk3W8QR+NSoEU+aKyQugoropb2qr2CAmq1yPfFkXxroQ5f5VyPcamSGvq70k08cSROeh0KO6KTWNcnrKIXCWM3bWuT9ZKgF+CStFGN4lcQ84GKHLk5KJeKOjYewsnERHBT26RPtL/1DhQ/ykAq+r0oCEcgEXcgCSIG/0OUwIUr/iHcKS1JVlrfTPTFNr7oLw1B/iUM5BtozORIFSuuxKPsrpiXFr8A9+UAG1XpqSWzZOkgy8wadD4cigghHaCQSAS0ZN0QIhuQW8Q0h0VPBEehlx30yI42NgolzQxKWsmUv/IAvl9FEVRNhN4mXKVXLLxpdE4kS66gVGZLiY3H6qu8zfnkFyMn75IspHIxjlTK94nNGRKq+r0gCJHGxECUo4dOlnU+HPLIePijj98AKqyFLW9IpHeEVnRgIlm0L3MGRze2m0YNS046BEofGFV8IlXdTL+iVmGXFC131EMTAqBAdsbj8NaDUNdaTJDd4h2LSOymVahlmN6yZFnMivSl3VgcRQe+whzVHPCMRZvMcUQq139pKHG2Y9QKSrcP7M84ESU6w9tcR8L3W68O4478OVJviaKtEKHrTq60PmARtatmytLtLvVPFYNv2XHve/DfuBXbcTyU7PGj7wexGmckPwtK7vsosV7/GnvO30eucDz0N4+VfG06WTGl/WZf2ojB6lfn+wZyni6nWBrs7EDzPbAodgjxgdBRRelvmRXhHC+ayjm3LgrtZVcf1H8K5jZkHRcQKQlSm1q1/37XfiGUJ/1hnqWKMcQN0qv1oQdD7qBDUlqpPUG40HwN1LTuDfAOBMNhgr5MjDRbJoLMpX43N708zioz0xEtIa7CI04K5+9eW6tEmq3z1V9ZTlJlNNFkAFKysQ4UTTDUMLxPVHP8pHwU2op0iLuWte/Swdd7a3HA1VEWeUPSI/LBXpqu9ExKiitAmXnqCkKY9C1gh3tYdvJ6XhDnC20A0rk7NYYSNbJ4iBPSemnF/Os0d6AdJi7pSzm/Xnrt2bBKZhZQtairrhQuhXRxTraurpXKCjApuWRx3g7uFmfX8ZGE77jrZazjiFXXAxsE8HZqmF0K6WadPy5a559v1k7UKUcDpwn80ZYc4JggYTA3fpJ4FlnS5mjqz6apKy5K6mfP3urQ133rS/GP8YqzqEmgUtA0aAsFjMkglNmBugJbLonypXCI/NLNSy0ACoNZt/KWsid54/n/sxvAy07SxE3djUXFje1IQ30MiONjIQtrNwQmMdiHt5dJkqx8d9v8pzwfiFVb+j3yBBhwpk1NI/svB2SH6mIBulfh7rrTElMY2+oe1Z8EYWyOv14B0thr4RvOmy529c6QYbbLDBBhtssMEGG2xQUtzuQhRMag5A69uDFY12UI/nzzhsn9/u7pwL+0WNd/IHdY67FAA3yJniDtVM3NuteFzSp1YFYBUX+CL7FYT9o9xmT3+DNq39jKMHVdjF37fUrrtjOIbW9qFgFP+f1xfE+Z+tViUDx5irw+wp7qErU2Pf5zppVffubvkzD7e3AIrwdltFTbe2tlu5xNnHsFFLcC2I31XYwyG1508Fd7tVPeaH0YKtP2VfbKcSDUqEPczbn+wp7sND2/Gfh8LetqsV/pYW520v6TNfPj8jasRCcsCRetuKu61ecO1X8VbZysFzefuc2VuVO70wb7stqpvPeS0PUJuW0MI9waFt71F7PlG3uMWNYwVvR9FMt4V4Lm/bos6ifXupMwvzhi6xFXV0nOtI7jhdjIHFjTL051XU7Rb6r3qXbr+Ctws0murevhCfnsfbeUvU2TbW3coTe2ZR3m7xnT2391DnuRbuAJMgELhdTtzQzLcvpF3MX3ogK3hD864+ZR5HKMobGhx36w52kIFK252ivKF2sFN+6jwQGQJrhcWtQpGOxQ3qJ9JXzgCs4O14tfAX5+0IjqUiCHrwqFkxKMgbnl/1IO5lN6/1QUsoPMD7Ic6peWKV/kOObaVdaiHeVoz9ebwJJvYkOFCQN6JOUmSLVwgcls7f6d2cTGGNxi4EqUMlZQDWgjd0S1PWoBhvB7E6xfFZbhCPTex2SoFuq+m9dDQnNABrwdt5hTchxXi7iNVJyo/PCITWKraRBDYishJFc1jg2HGvB2+tf8jbQYW2jDg+q2QkBBjYWlXZy1cSocXAd4DQIjIA68HbP5W3VEDG50oZI2LDFc7LYo2PPZjIABThbeuTEPvkxv579g1Hl7GAYfETxWcJ0CAY4cFOlk4j0gImMACFeBNnC60X4+23IEApwhs3Hyx/gvgsPaUqNQr+pLRBw8krYwAK8SZG5aV4O0hisAQFeEupk0REp5p71lGKbLuVFtLdSto68Qnam/NW3bUPGJx/xglmahAFeBPY64u0ZxSgyrL9u5q++j4XsPEJWiHexNW3v8mln8Xb1jGLVqWKr5GSwwK8CeID7BrzBY71Hlhoq5TQRgkCE+NR4TVGIX+6I8QuGd3zeBOjmvaCq3nD6rTFSi8XivE4QD0TP/DE+Qk85t+MWuD7QZX33jwOydB5LlpdzRsuIG21GOB91cLlJJsT2qheuc10i28RdT/Wg7c9vt64kjeqHstLb379Egsc6pv3AYc5Be5E4F6Ztz2etyovH6Kq2Ure9nPmt7WXK3CUteKS9vMcY0JVBF6CNyT3FVG0iYuJxHhgf3rE2CO8OLDNn7yKN2xxBKElFozc+uV5XIvcaaUneFEVdxsZANLsJXjDfHDFGSkqNJC0Wxi/RQUKniAcUWTnmpjX4z0eSJiKlJPuInHjC0jbgm6x3yVNVxQmjzOkQTR3voCDpSnWQmHci70+P09svPeOeD+OKD7PLHlLWOBy65eRbfx8h5WSuml3XDRHgAvmZJz47hzu8sNDh1HZZfuT4HDShqzEVQ4/0+12n/5g4Y6NqThfeMIh15/U7qiuX80IGy+yM6oi5SRsHPHCK19AEq+xYoGLhAAvvGzzUS2OHe+yRk+vO5OluO0q0y5aDk6GlZFnYYuS1vODvzNsM4r02RiMRf76MsYOvXhIiRufPCRg0pPdLG+OC+p5zp7yWhfZy6zVxLVl5ad4pTJtyn9n9Il4y1nRe47AoQHSQXIlvoIAzALGpwy/Gy1EHOa55YSSu4xl/e3Kp+RuZtZDjrH0pCTkDkisqAyzAwtGaCvD7h5sMeUaIc73KlF3NL9PcGc1y00eteDRqP3BYUs4vMhY2n/Eo0egoqSdT4J21dYeLURHcFQtUd0yap5yquef/2SU/Y724VZmvegzapif3dtHqPMLZjiH6AJZKxQ2Pkzu1u3doagseZB/GOKQ9sPnR6mGhxdPrEDs7meNagfxkE3EBhtssMEGG2ywwQavgPZyuXnPQQ4GEzE9l5pV9i+BvCqGqvDTFQOtpN8ieysEhog3P3BL/+rL14WYt3V7zYEN3y1FFCD0YguSbKKXSzEq0obn0K8oQ6+fwpvgUEi2fMoehfR10IVRv9RB0mFgztrcJbnT/11ML8fwTWVjF7/levkj/uBGD2+GozF+l9m4E1nrcGBo6JxO9DZBfwGbjOHbLOeLsQ7aBkvJvwSNdM3BLxGxZzI6R3O6ETveKTxJC+B2d4g7dPEVAtmKLnlJHMQywC3W5Z0kA83SHdcd6qaGXi7T04cxbwbanBh4Eir4H6lPOzAMfeG6sh7t8HRHj3ibg03ZXRiWNgPNA9Cvo87QSaquB64b6OBUJJe2bBkRb+El2BvgQaAXYhLeVNgPGsupbqIWqqWuxTtw5qrVmYe2HfoTdQyHyPFmG9YIvYLN6y0cA2rVqWV0Pfi2trlrqT7YMTGHc9gCtL605CnozhtZluP6IejXdZDUtjs9D17Hm+kmekfXUreWqF9J6hvGDB30Ryb6niLQ05BccgxZnunGhLRYi6/wzoyAGJVeFwoPx5unxp988XT4ynrwLxm5HZjwMwtu/Ommdnxs4TjYcvm6zgZjPUODR/oG+S5s6Jjxa1ddzZFov+CZqEPTit9oOzLd/2G+L4WZgeQjeXVgHm+SZnbhp4qSdwjOzIUEeSMfqQAkRXa+H09Pi6j0l4M+goVYSXibxycBDttwM+HNduFmW9fj97JODWcNohFfdQz89krVHUDqVvPWNZwOwUIeSyxv5NN+M4PlzV9oegTZguKV8NYz1FSIkebN151LcsnL9fh64PxSwyZYN9FMe4a8kjc5fk+oqv9HKsRbOLSs0QAjQEr3PN5kPbnkj3XgDaoGMsHziQVfUjzXNaIGGbwtDd2nIBXibaobsaZ1TOgSKd70VbxBw0pdcg0CYDxxhNCBBpyiaWZA48Xwhm4+tQObxQK89QwzvualycobuELMaT8YSjxvtkp9V8teA9qk2Q8tGqI9MFFQ4VineGDeEM0v4c0eoLDDDiySJ861H3DyBXib62Y8c9dg7Zvkxh1ODfQFrTRvIPQZRoPwgrG1BsTZC2DW+t1uF76ZHE1kqVmLAdwxxKGTpzqdU4SFib9vtVQdeQRaDDqGheo6BXizF5Z5ibvpOChGo3gDQSTqcAY7hNaV4w1E1ubpDLQ4dZz1+Dpq2CF+Tpvg+zgYE8eHBuipsmkg6KqLXcaStFBdxNIi/l6APx4T3rRFtO+HCiXNW4CEAfejIkb7asybNDdAwoA6XKDTh1rM2wInDL6Dr2jo5lrQBuB1J6PRqL9sp3b0InVVnQn2g0uftLB7fdBiMot2zPpEB9v9PtG4/iza1+9j6wXiN4QZnnivT9c8pgPQ4Yh0OOgTcuwZ2ZzPUIvU+/XXF7Rf2KA4Nrz9M7xP3v4LNjJSovOJ4X8AAAAASUVORK5CYII=", width=500)
    with col3:
        st.write("")


    st.subheader("Overview".upper())
    st.write("- Monthly sales analysis")
    st.write("- Suppliers analysis: which ones contribute to the highest revenues? What are their top products?")
    st.write("- Customers analysis: what segment of customers contributes to the highest revenue? What is their purchasing power?")
    st.write("- Recommendation system to increase revenues")

    # check data
    data_button = st.button("Check Data")
    if data_button:
        data = df_customers2019.head()
        data


# Sales Page
if add_selectbox == 'Sales':
    st.title(" Monthly Sales")
    Total_sales = df_customers2019['Operation Net Total'].sum()
    st.write("TOTAL SALES IN SEPTEMBER 2019:",int(Total_sales), "LBP")
    df=df_customers2019.groupby(['Date','Unit-Measure'],as_index=False)['Operation Net Total'].sum()



    if st.checkbox("Sales By Category"):
        st.subheader("Sales By Category")
        # Filter by category
        opt = st.selectbox(
        'Choose Category',
        df_customers2019['Unit-Measure'].unique()
        )
        df_filtered = df[df['Unit-Measure']== opt]
        sales_by_category = df_filtered[df_filtered['Unit-Measure']== opt]['Operation Net Total'].sum()
        st.write(opt,"Sales:", int(sales_by_category),"LBP")

        # plot daily sales
        st.subheader("Daily Sales By Category")
        fig1 = px.line(df_filtered, x='Date', y="Operation Net Total", color='Unit-Measure')
        fig1


    if st.checkbox("Sales Pattern"):

        #st.header("Daily Sales")
        st.subheader("Sales Pattern")

        # check sales pattern by category
        fig0 = px.line(df, x='Date', y="Operation Net Total", color='Unit-Measure')
        fig0


# Customer Analysis Page
if add_selectbox == 'Customer Analysis':
    # Study
    st.title("Customer Analysis")

    #st.subheader("Customer Statistics")
    df_customers2019=df_customers2019.groupby(['Customer Description'],as_index=False)['Operation Net Total'].sum().sort_values('Operation Net Total',ascending = False)
    # Define dataframe Client Divers
    Client_divers = df_customers2019[df_customers2019['Customer Description'] == 'CLIENT DIVERS']
    # Define dataframe loyal custumers
    loyal_customer = df_customers2019[df_customers2019['Customer Description'] != 'CLIENT DIVERS']

    # Display Revenue by customers segment
    option = st.selectbox(
    'Choose Customer Segment'.upper(),
    [ 'LOYAL CUSTOMERS','CLIENT DIVERS','ALL CUSTOMERS']
    )

    if option =='CLIENT DIVERS':
        revenue = Client_divers['Operation Net Total'].sum()
        st.write( 'Revenue by', option, 'is', int(revenue),'LBP')

    elif option == 'LOYAL CUSTOMERS':
        # revenues made by loyal customers
        revenue = loyal_customer['Operation Net Total'].sum()
        st.write( 'Revenue by', option, 'is', int(revenue),'LBP')
        # Nb of loyal customers
        st.write ('Number of Loyal Customers:', len(loyal_customer))
        # Average purchase power
        st.write( 'Average Purchase Power:',int( revenue/len(loyal_customer)),'LBP')

        #How many are driving the most of the Revenues
        st.subheader("Distribution of Loyal Customers According to Purchase Power")
        segment1 = len(loyal_customer[loyal_customer['Operation Net Total'] < 200000])
        segment2 = len(loyal_customer[(loyal_customer['Operation Net Total'] >= 200000) & (loyal_customer['Operation Net Total'] < 600000)] )
        segment3 = len(loyal_customer[loyal_customer['Operation Net Total'] >= 600000])
        dict = {"Segment":["segment1:<200000 per month", "segment2:>=200000 & < 600000 per month","segment3:>=600000 per month"], "Values":[segment1,segment2,segment3]}
        df_segments = pd.DataFrame(dict)
        # plot pie chart
        fig = px.pie(df_segments, values = 'Values', names = 'Segment')
        if st.checkbox("Display Chart"):
            fig
        # Check Top Ten loyal customer
        st.subheader("Top 10 Loyal Customers")
        if st.checkbox("Display List"):
            y=loyal_customer.head(10)
            y

    elif option == 'ALL CUSTOMERS':
        revenue = df_customers2019['Operation Net Total'].sum()
        st.write( 'Revenue by', option, 'is', int(revenue),'LBP')



# Supplier Analysis Page
if add_selectbox == 'Supplier Analysis':
    # Supplier comparison for loyal customers and clients divers
    st.title("Supplier Analysis")
    st.subheader("Suppliers Comparison by Section ")

    section = st.selectbox(
    'Choose section'.upper(),
    df_customers2019.Section.unique()
    )

    Suppliers1 = Client_divers.loc[Client_divers.Section == section].groupby(['Brand'],as_index=False)['Operation Net Total'].sum()
    Suppliers2 = loyal_customer.loc[loyal_customer.Section == section].groupby(['Brand'],as_index=False)['Operation Net Total'].sum()

    CD = go.Bar(x=Suppliers1.Brand,
                y=Suppliers1['Operation Net Total'],
                name='Client Divers',
                marker=dict(color='#56606D'))


    LC = go.Bar(x=Suppliers2.Brand,
                y=Suppliers2['Operation Net Total'],
                name='Loyal Customers',
                marker=dict(color='#59903D'))


    layout = go.Layout(title="Revenues by Customer Segments"
                    )

    fig = go.Figure(data=[CD,LC], layout=layout)
    if st.checkbox("Display Chart"):
        fig

    # item analysis
    st.subheader("Quantity Sold by Supplier/Item")
    Brand = st.selectbox(
    'Choose Brand'.upper(),
    df_customers2019.Brand.unique()
    )
    brand1 = Client_divers.loc[Client_divers.Brand == Brand].groupby(['itemdescription1'],as_index=False)['Tot Qty'].sum().sort_values('Tot Qty',ascending = True)
    brand2 = loyal_customer.loc[loyal_customer.Brand == Brand].groupby(['itemdescription1'],as_index=False)['Tot Qty'].sum().sort_values('Tot Qty',ascending = True)

    CD_item = go.Bar(y=brand1['itemdescription1'],
                x=brand1['Tot Qty'],
                name='Client Divers',
                marker=dict(color='#56606D'), orientation='h')
    LC_item = go.Bar(y=brand2['itemdescription1'],
                x=brand2['Tot Qty'],
                name='Loyal Customers',
                marker=dict(color='#56608D'),orientation='h')

    layout1 = go.Layout(title="Preferred Items by Client Divers",
                    yaxis=dict(title='Item'),
                    xaxis=dict(title='Qty')
                    )

    layout2 = go.Layout(title="Preferred Items by Loyal Customers",
                    yaxis=dict(title='Item'),
                    xaxis=dict(title='Qty')
                    )

    fig1=go.Figure(data=CD_item, layout=layout1)
    fig2=go.Figure(data=LC_item, layout=layout2)
    if st.checkbox("Display Charts"):
        fig1
        fig2

# Recommendation System
if add_selectbox == 'Recommendation System':
    st.title("Recommendation System")
    expander = st.beta_expander("Overview".upper())
    expander.write("""This recommendation system aims at enhancing the management
    desicions concerning targetted promotions.
    It studies the purchasing behavior of loyal customers to grow their share of wallet (Market Basket Analyis)
    and proposes solutions to increase sales (i.e. sales of products with close expiry date).""")


    # Association rule mining
    
    # dropping first column
    df1 = df_encoding.drop(['Item(s)'], axis=1)

    # getting uniques Items
    columns = df1.columns.to_list()
    column_values = df_encoding[columns].values.ravel()
    items =  pd.unique(column_values)


    # Manual Encoding
    itemset = set(items)
    encoded_vals=[]
    for index, row in df1.iterrows():
        rowset=set(row)
        labels={}
        uncommon=list(itemset-rowset)
        common=list(rowset)
        for uc in uncommon:
            labels[uc]=0
        for c in common:
            labels[c]=1
        encoded_vals.append(labels)
    onehot_df = pd.DataFrame(encoded_vals)
    onehot_df = onehot_df.loc[:, onehot_df.columns.notnull()]

    # get frequent items
    #freq_items=apriori(onehot_df, min_support=0.006, use_colnames=True,verbose=1)
    
#     def freq_itms(onehot_df):
#         freq_items =apriori(onehot_df, min_support=0.006, use_colnames=True,verbose=1)
#         return freq_items
    
    freq_items = freq_itms(onehot_df)
#     #get association rules
#     rules=association_rules(freq_items, metric="confidence", min_threshold=0.25)
#     # sorting by confidence
#     rules=rules.sort_values(by=['confidence'], ascending=False)
#     #rules_sorted_by_conf

#     #transform items array to a list
#     items = items.tolist()
#     items.pop(4)

#     # define length_antecedents, length_pconsequents
#     rules['length_antecedents'] = rules['antecedents'].apply(lambda x: len(x))
#     rules['length_consequents'] = rules['consequents'].apply(lambda x: len(x))



#     if st.checkbox("Increase Sales by Item"):
#     # Recommendation System 1
#         st.title("Increase Sales by Item")
#         item = st.selectbox(
#         'Select Item'.upper(),
#         items
#         )

#         # define consquent item to get the itemset
#         rules_item=rules[rules['consequents']=={item}]
#         X = rules_item.sort_values(by=['confidence'], ascending=False)
#         st.header("Proposed Solution")
#         st.write("Provide discounts on the below itemsets")
#         X['antecedents']

#     if st.checkbox("Grow Share of Wallet"):
#         # Recommendation System 2
#         st.title("Grow Share of Wallet")
#         st.subheader("Define Basket")
#         # create a list with possible lenghth of item set
#         nb_of_antecedents = st.selectbox("Select number of items".upper(), rules['length_antecedents'].unique())

#         # Display items
#         antecedents = set()
#         for i in range(nb_of_antecedents):
#             itemi = st.selectbox(
#             " Select item ".upper()+str(i+1),
#             items
#             )
#             antecedents.add(itemi)

#         st.write('Items in Basket:',antecedents)
#         st.header("Proposed Solution")
#         st.write("Provide discounts on the below items")
#         predicted_item = rules[rules['antecedents']== antecedents]
#         predicted_item['consequents']
