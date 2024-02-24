import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score,mean_absolute_error
import joblib

df = pd.read_csv('altered_data.csv')

age_group = {
    'less than 18':0,
    '18-25':1,
    '25-40':2,
    '40-60':3,
    'more than 60':4
}
ordered_from = {
    'hungerstation':0,
    'jahez':1
}
cuisine = {
    'breakfast':0,
    'sandwich':1,
    'healthy':2,
    'pizza':3,
    'traditional':4,
    'coffee':5,
    'dessert':6,
    'fastfood':7
}
t_branches = [
       'alsalam', 'Al Naseem District West', 'Ghirnatah District',
       'Al Malga District', 'Al Aqeeq District', 'Rayyan District',
       'Al Nada District', 'Qurtubah', 'As Sulaymaniyah District',
       'King Abdullah District', 'Al Ishbiliyah District',
       'Hamra District', 'Al Yasmin District',
       'As Suwaydi al Gharbi District', 'Al Rawdah District',
       'Al Rabiea District', 'Al Uarayja al Gharbiyah District',
       'Hittin District', 'As Sahafah District', 'Al Rimal District',
       'Al Olaya District', 'Al Wuroud District', 'Al Nakheel District',
       'Hijrat Laban', 'As Salamah', 'North Maathar District',
       'Al Badiah District', 'Al Masif District', 'Al Izdihar District',
       'At Taawun Distrikt', 'Al Muhammadiyah District',
       'Al Rawabi District', 'Zahrat Laban District',
       'Al Nahdhah District', 'King Fahd District', 'Al Nafil District',
       'King Faisal District', 'Al Fakhiriyah District',
       'Al Mugharzat District', 'Mursalat District',
       'Al Madhar ash Shamali District', 'Al Qouds District',
       'Almuruj Dstrict', 'Al Fayha District', 'Al Raid District',
       'Al Ghadir District', 'Zahrat al Badiah District', 'An Nasim',
       'Al Jazirah District', 'Al Naseem District East',
       'Al Safarat District', 'Tuwayq District', 'Al Falah District',
       'Al Mansurah District', 'Al Yarmouk District',
       'Al Janadriyah District', 'Ad Dar Al Baida District',
       'Al Hazm District', 'Jarir District', 'Al Mutamarat District',
       'Ad Dubbat District', 'Al Waha District', 'umu alhamaam alsharqii',
       'Al Nuzhah District', 'Al Aziziyah District', 'Shubra District',
       'Irqah District', 'Al Rabwah District', 'Al Muwanisiyah',
       'As Salihiyah District', 'An Nakhil', 'Ash Shifa District',
       'District of Umm Al-Hamam (West)', 'Badr District',
       'Okaz District', 'Umm Salim', 'Al Nargis District',
       'Al Andalus District', 'Al Malaz District',
       'Ar Rahmaniyah District', 'Al Wadi District',
       'Al-Qadisiyah District', 'Al Zahraa District',
       'Al Murabba District', 'As Safah District', 'Al Masani District',
       'Al Zahrah District', 'Diplomatic Quarter',
       'Al Urayja al Wusta District', 'Turayfah', 'Salah ad Din District',
       'mahdia District', 'Dhrat Nemar', 'Al Khalij District',
       'As Sa`adah District', 'Al-Nazim District', 'Sultanah District',
       'As-Suwaidi District', 'An Narjis', 'An Namudhajiyah',
       'Al Marwah District', 'Al Mu`ayzilah', 'Al Khuzama District',
       'Siyah District', 'Ash Shumaysi District', 'Ad Diriyah Al Jadida',
       'Al Washam District', 'Al Manar', 'Al Manar District',
       'Ash Sharafiyah', 'Al Faruq District', 'Al Nour District',
       'Al Oud District', 'Imam Muhammad ibn Saud University',
       'Al Malik Faysal', 'alisha', 'Ad Duraihimiyah District',
       'Al Hada District', 'Namar District', 'Manfuhah District',
       'Al Rimayah District', 'Al Rafiah District', 'Al Marqab District',
       'Al Deerah District', 'King Salman Park', 'Gbraya District',
       'Al Futah District', 'Al Wizarat District', 'aleatiqa', 'Ad Dahu',
       'Dirab', 'Manfuha Al-Jadidah District'
]
branches = dict(zip(t_branches,range(len(t_branches))))

t_customer_district = [
       'As Sulayy District', 'Al Manar', 'Al Muwanisiyah',
       'Al Malga District', 'As Sahafah District', 'Al Nahdhah District',
       'Al Rabiea District', 'Qurtubah', 'Al Rabwah District',
       'Al Masif District', 'Al Yarmouk District', 'Hamra District',
       'Al Yasmin District', 'Dhrat Nemar', 'Al Andalus District',
       'Al Aqeeq District', 'Al Uarayja al Gharbiyah District',
       'At Taawun Distrikt', 'Al Rawdah District', 'Al Nakheel District',
       'As Sulaymaniyah District', 'Al Ishbiliyah District', 'alsalam',
       'Al Nafil District', 'Al Rimal District', 'Al Olaya District',
       'Suburb of Aljwan', 'King Fahd District', 'Zahrat Laban District',
       'Ar Rahmaniyah District', 'District of Umm Al-Hamam (West)',
       'Al Fakhiriyah District', 'Al Ghadir District',
       'Mursalat District', 'Al Nuzhah District', 'Al Fayha District',
       'Irqah District', 'Ash Shifa District', 'Al Naseem District East',
       'Al Urayja', 'Al Nada District', 'Al Rimayah District',
       'Al Badiah District', 'Al Izdihar District', 'Almuruj Dstrict',
       'Ghirnatah District', 'Al Qouds District', 'Tuwayq District',
       'Al-Qadisiyah District', 'Al Naseem District West',
       'Al Khuzama District', 'Al Malaz District', 'Al Wuroud District',
       'Hittin District', 'Al Muhammadiyah District',
       'Al Mutamarat District', 'Al Zahraa District', 'Rayyan District',
       'Al Waha District', 'Al Manakh District', 'Al Janadriyah District',
       'Al-Nazim District', 'Al Mansurah District', 'Badr District',
       'Al Khalij District', 'As Salamah', 'Al Jazirah District',
       'King Faisal District', 'Turayfah', 'Al Aziziyah District',
       'Salah ad Din District', 'Zahrat al Badiah District',
       'North Maathar District', 'As Sinaiyah District',
       'As Suwaydi al Gharbi District', 'Al Wadi District',
       'Hijrat Laban', 'Al Mugharzat District', 'Al Rafiah District',
       'Al Safarat District', 'An Nakhil', 'Ad Diriyah Al Jadida',
       'An Namudhajiyah', 'Al Rawabi District', 'Shubra District',
       'Ad Dar Al Baida District', 'Al Nargis District',
       'Al Hazm District', 'Al Madhar ash Shamali District',
       'Al Murabba District', 'Al Mu`ayzilah', 'King Abdullah District',
       'Al Khalideyeh District', 'Khashm al `An', 'As Sa`adah District',
       'Al Falah District', 'An Nasiriyah District', 'Ad Dahu', 'Askan 3',
       'An Nasim', 'Manfuha Al-Jadidah District', 'Sultanah District',
       'alsharq', 'Imam Muhammad ibn Saud University',
       'umu alhamaam alsharqii', 'Al Washam District',
       'Al Manar District', 'Al Urayja al Wusta District',
       'Al Zahrah District', 'Al Raid District', 'Ash Sharafiyah',
       'Ad Dubbat District', 'Diplomatic Quarter', 'As-Suwaidi District',
       'Al Malik Faysal', 'Al Jarradiyah District', 'Al Amal',
       'As Safah District', 'Al Wizarat District', 'An Narjis',
       'Al Faisaliah District', "Ad Difa' District", 'An Nazim', 'alisha',
       'Okaz District', 'Namar District', 'King Salman Park',
       'Jarir District', 'Al Jazah', 'Al Nour District',
       'Al Hada District', 'Al Qadisiyah', 'Al Marwah District',
       'Ad Duraihimiyah District', 'Al Iskan District',
       'Al Futah District', 'Al Fawwaz district',
       'Al Maizliyyah District', 'As Skirinah District',
       'Al Yamamah District', 'Wadi Laban', 'mahdia District',
       'Mahdia District', 'Manfuhah District', 'Dirab',
       'Ash Shumaysi District', 'Gbraya District', 'Al Wusayl',
       'aleatiqa', 'Dahiat Namar', 'Siyah District', 'ahad', 'Umm Salim',
       'As Salihiyah District', 'Jabrah District', 'Al Oud District',
       'At Turayf Al Jadid', 'Thulaym District', 'Al Masani District',
       'Mikal District', 'Al Marqab District', 'Salam District',
       'Al Faruq District', 'Askan 1', 'Al Muhammadiyah',
       'Riyadh housing', 'Umm Salim District', 'Al Mashael District',
       'Banban', 'Al Deerah District', 'Al Muayzilah',
       'Al Wasitiyya District', 'Al Ghannamiyah', 'Ad Dubiyah District'
]
customer_district = dict(zip(t_customer_district,range(len(t_customer_district))))

df['age_group'] = df['age_group'].replace(age_group)
df['ordered_from'] = df['ordered_from'].replace(ordered_from)
df['cuisine'] = df['cuisine'].replace(cuisine)
df['branch_district'] = df['branch_district'].replace(branches)
df['customer_district'] = df['customer_district'].replace(customer_district)

new_df = df.drop(['Order_ID','date_time','branch_id','order_to_deliver','estimation','hour','minute','day_name','distance'],axis=1)

X = new_df.drop(['ordered_from','age_group','order_price'],axis=1)
y = new_df['age_group']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

app_model = LogisticRegression()

app_model.fit(X_train, y_train)

y_pred = app_model.predict(X_test)

joblib.dump(app_model, 'application_model.pkl')

X = new_df.drop(['ordered_from','age_group','order_price'],axis=1)
y = new_df['age_group']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

age_model = LogisticRegression()

age_model.fit(X_train, y_train)

y_pred = age_model.predict(X_test)

joblib.dump(age_model, 'age_group_model.pkl')

X = new_df.drop(['ordered_from','age_group','order_price'],axis=1)  
y = new_df['order_price'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

price_model = LogisticRegression()

price_model.fit(X_train, y_train)

y_pred = price_model.predict(X_test)

joblib.dump(price_model, 'order_price_model.pkl')