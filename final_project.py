#import ajkna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

real_estate = pd.read_csv('assets/real_estate.csv', sep=';')
#print(real_estate.head())
#print(real_estate.describe())

row_most_exp = np.argmax(real_estate['price'])
print('Most expensive home is:', real_estate.loc[row_most_exp, 'realEstate_name'], 'which costs:', real_estate.loc[row_most_exp, 'price'])

row_cheapest = np.argmin(real_estate['price'])
print('Cheapest home is:', real_estate.loc[row_cheapest, 'realEstate_name'], 'which costs:', real_estate.loc[row_cheapest, 'price'])

row_biggest = np.argmax(real_estate['surface'])
print('Biggest home is:', real_estate.loc[row_biggest, 'realEstate_name'], 'which has a surface of:', real_estate.loc[row_biggest, 'surface'])

row_smallest = np.argmin(real_estate['surface'])
print('Smallest home is:', real_estate.loc[row_smallest, 'realEstate_name'], 'which has a surface of:', real_estate.loc[row_smallest, 'surface'])

diff_level5 = len(real_estate.groupby('level5').sum())
print('There are', diff_level5, 'different populations (level5)')

print('Does the dataset contain NAs?:', real_estate.isnull().values.any())

data_frame_no_nulls = real_estate.dropna()
print(data_frame_no_nulls)

#level5_arroyomolinos = real_estate.loc[real_estate['level5']=='Arroyomolinos (Madrid)']
level5_arroyomolinos_prices = real_estate.loc[real_estate['level5']=='Arroyomolinos (Madrid)']['price']
mean_price_arroyomolinos = level5_arroyomolinos_prices.mean()
print('Mean price in the population (level5) of Arroyomolinos (Madrid):', int(mean_price_arroyomolinos))

#plt.hist(level5_arroyomolinos_prices)
plt.hist(level5_arroyomolinos_prices, 10, histtype = 'bar')
plt.ylabel("Price")
plt.xlabel("Bin Number")
plt.title("Prices distribution in Arroyomolino")
plt.show()

madrid_south_belt_cities = sorted(['Fuenlabrada', 'Leganés', 'Getafe', 'Alcorcón'])
madrid_south_belt = real_estate[real_estate['level5'].isin(madrid_south_belt_cities)]

median_price_mad_south_belt = madrid_south_belt.groupby('level5')['price'].median()
print("Median price for each population in Madrid's South Belt:\n".upper(), median_price_mad_south_belt)

median_price_mad_south_belt.plot(kind='bar')

max_price_mad_south_belt = madrid_south_belt.groupby('level5')['price'].max()
print("Maximum price for each population in Madrid's South Belt:\n".upper(), max_price_mad_south_belt)

columns = ['price', 'rooms', 'surface', 'bathrooms']
mean_data_mad_south_belt = madrid_south_belt.groupby('level5').mean().filter(columns)
print("Mean price, rooms, surface and bathrooms for each population in Madrid's South Belt:\n".upper(), mean_data_mad_south_belt)

variance_data_mad_south_belt = madrid_south_belt.groupby('level5').var().filter(columns)
print("Variance of price, rooms, surface and bathrooms for each population in Madrid's South Belt:\n".upper(), variance_data_mad_south_belt)

columns = ['realEstate_name', 'price']
most_exp_mad_south_belt = madrid_south_belt.groupby('level5')['realEstate_name', 'price'].max()
print("Most expensive home for each population in Madrid's South Belt:\n".upper(), most_exp_mad_south_belt)

#normalizar(precio) = (precio - precio_min) / (precio_max - precio_min)
madrid_south_belt_price_normal = madrid_south_belt[['level5', 'price']]
madrid_south_belt_price_normal['price'] = ( (madrid_south_belt['price'] - madrid_south_belt['price'].min()) / (madrid_south_belt['price'].max() - madrid_south_belt['price'].min()) )
#print(madrid_south_belt_price_normal)
#madrid_south_belt_price_normal_by_pop = madrid_south_belt_price_normal.groupby('level5')

#Split DataFrame based on population:
population_specific_data_frames = [
    madrid_south_belt_price_normal[madrid_south_belt_price_normal['level5'] == population]
    for population in madrid_south_belt_cities
]

for index, population_data_frame in enumerate(population_specific_data_frames) :
    plt.subplot(2, 2, index + 1)
    plt.xlabel('Normalized price')
    plt.ylabel('Frequency')
    plt.hist(population_data_frame, alpha=0.5,
        label=population_data_frame['level5'].iloc[0])

plt.show()

# Price per m2 in Getafe:
getafe_homes = madrid_south_belt[madrid_south_belt['level5'] == 'Getafe'][['price', 'surface']]
getafe_prices_per_m2 = getafe_homes['price'].mean() / getafe_homes['surface'].mean()
print("Getafe's mean price per sq meter is", int(getafe_prices_per_m2))

# Price per m2 in Getafe:
alcorcon_homes = madrid_south_belt[madrid_south_belt['level5'] == 'Alcorcón'][['price', 'surface']]
alcorcon_price_per_m2 = alcorcon_homes['price'].mean() / alcorcon_homes['surface'].mean()
print("Alcorcón's mean price per sq meter is", int(alcorcon_price_per_m2))

# Valdemorillo
valdemorillo_homes = real_estate[real_estate['level5'] == 'Valdemorillo'][['price', 'surface']]
valdemorillo_mean_price = valdemorillo_homes['price'].mean()
valdemorillo_price_per_m2 = valdemorillo_mean_price / valdemorillo_homes['surface'].mean()

# Galapagar
galapagar_homes = real_estate[real_estate['level5'] == 'Galapagar'][['price', 'surface']]
galpagar_mean_price = galapagar_homes['price'].mean()
galpagar_price_per_m2 = galpagar_mean_price / galapagar_homes['surface'].mean()

print("Valdemorillo's mean price is:", int(valdemorillo_mean_price) , "and mean price per sq meter is" , int(valdemorillo_price_per_m2))
print("Galpagar's mean price is:", int(galpagar_mean_price), "and mean price per sq meter is" , int(galpagar_price_per_m2))

# price-surface scatter plot
plt.scatter(valdemorillo_homes['surface'], valdemorillo_homes['price'], label='Valdemorillo')
plt.scatter(galapagar_homes['surface'], galapagar_homes['price'], label='Galapagar')
plt.legend(loc='upper left')
plt.ylabel('Surface')
plt.xlabel('Price')
plt.title("Valdemorillo and Galpagar: prices vs surface")
plt.show

# 4 price-surface scatter plots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22,8))
fig.suptitle('Comparation of price vs surface')
fig.supxlabel('Surface')
fig.supylabel('Price')
fig.set_figwidth

ax1.scatter(getafe_homes['surface'], getafe_homes['price'], label='Getafe')
ax1.legend(loc='upper left')

ax2.scatter(alcorcon_homes['surface'], alcorcon_homes['price'], label='Alcorcón', c='red')
ax2.legend(loc='upper left')

ax3.scatter(valdemorillo_homes['surface'], valdemorillo_homes['price'], label='Valdemorillo', c='black')
ax3.legend(loc='upper left')

ax4.scatter(galapagar_homes['surface'], galapagar_homes['price'], label='Galpagar', c='green')
ax4.legend(loc='upper left')

#Real Estate agencies:
diff_agencies = len(real_estate.groupby('id_realEstates').sum())
print('There are', diff_agencies, 'real estate agencies')

homes_per_population = real_estate.groupby('level5').count()
print(homes_per_population['price'])

