# Forecasting_Sticker_Sales-Regression-AI-Model

# Final Version

## Class Definition & Initialization (__init__)
The `DateFeatureEngineer` class is designed to extract and enhance date-related features in a dataset. The constructor (`__init__`) takes a `DataFrame` (df) and makes a copy of it to avoid modifying the original data.

- Country Specific Holiday Dictionary (`country_holidays`)
  - Defines major holidays for six countries: Canada, Finland, Italy, Kenya, Norway, and Singapore.
  - Holidays are stored as tuples of (month, day).
- GDP Dictionary (`gdp_dict`)
  - Stores GDP values for the same six countries from 2010 to 2020.

```python
class DateFeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        self.country_holidays = {
            'Canada': [(1, 1), (7, 1), (12, 25), (12, 26)],
            'Finland': [(1, 1), (12, 6), (12, 25), (12, 26)],
            'Italy': [(1, 1), (8, 15), (12, 25), (12, 26)],
            'Kenya': [(1, 1), (6, 1), (12, 12), (12, 26)],
            'Norway': [(1, 1), (5, 17), (12, 25), (12, 26)],
            'Singapore': [(1, 1), (8, 9), (12, 25)],
        }
        self.gdp_dict = {
            "Canada": {2010: 1617.34, 2011: 1793.33, 2012: 1828.37, 2013: 1846.60, 2014: 1805.75,
                        2015: 1556.51, 2016: 1527.99, 2017: 1649.27, 2018: 1725.33, 2019: 1743.73, 2020: 1655.68},
            "Finland": {2010: 249.42, 2011: 275.60, 2012: 258.29, 2013: 271.36, 2014: 274.86,
                        2015: 234.53, 2016: 240.77, 2017: 255.65, 2018: 275.71, 2019: 268.51, 2020: 271.89},
            "Italy": {2010: 2136.10, 2011: 2294.99, 2012: 2086.96, 2013: 2141.92, 2014: 2162.01,
                      2015: 1836.64, 2016: 1877.07, 2017: 1961.80, 2018: 2091.93, 2019: 2011.30, 2020: 1897.46},
            "Kenya": {2010: 45.41, 2011: 46.87, 2012: 56.40, 2013: 61.67, 2014: 68.29,
                      2015: 70.12, 2016: 74.82, 2017: 82.04, 2018: 92.20, 2019: 100.38, 2020: 100.66},
            "Norway": {2010: 431.05, 2011: 501.36, 2012: 512.78, 2013: 526.01, 2014: 501.74,
                       2015: 388.16, 2016: 370.96, 2017: 401.75, 2018: 439.79, 2019: 408.74, 2020: 367.63},
            "Singapore": {2010: 239.81, 2011: 279.36, 2012: 295.09, 2013: 307.58, 2014: 314.86,
                          2015: 308.00, 2016: 319.05, 2017: 343.26, 2018: 376.89, 2019: 376.90, 2020: 349.49},
        }
```

## Function Definitions
The class provides multiple methods for feature extraction:

### extract_date_features
- Converts the `date` column into a datetime format.
- Extracts date components: year, month, day, day of the week, week of the year, and day of the year.
- Creates binary indicators:
  - is_weekend: Checks if the day is Saturday (5) or Sunday (6).
  - is_holiday: Checks if the date is a holiday for the respective country.
- Adds season classification based on month.
- Defines holiday_period to mark a broader holiday season (±35 days from major holidays).
- Applies cyclical encoding using sine and cosine transformations:
  - day_of_week_sin & day_of_week_cos
  - month_sin & month_cos
  - day_of_year_sin & day_of_year_cos
- Fetches GDP values and computes the GDP percentage change from the previous year.
- Drops unnecessary columns: date, month, day, and week_of_year.

```python
def extract_date_features(self):
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
        self.df['day_of_year'] = self.df['date'].dt.dayofyear
        self.df['is_weekend'] = self.df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        self.df['is_holiday'] = self.df.apply(lambda row: (row['month'], row['day']) in self.country_holidays.get(row['country'], []), axis=1).astype(int)
        self.df['season'] = self.df['month'].apply(self.assign_season)
        self.df['holiday_period'] = self.df.apply(lambda row: 1 if row['is_holiday'] == 1 else self.flag_holiday_period(row), axis=1)
        self.df['day_of_week_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_of_week_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['day_of_year_sin'] = np.sin(2 * np.pi * self.df['day_of_year'] / 365)
        self.df['day_of_year_cos'] = np.cos(2 * np.pi * self.df['day_of_year'] / 365)
        self.df['gdp'] = self.df.apply(lambda row: self.gdp_dict.get(row['country'], {}).get(row['year'], None), axis=1)
        self.df['gdp_change'] = self.df.apply(lambda row: self.calculate_gdp_change(row), axis=1).fillna(0)
        return self.df.drop(columns=['date', 'month', 'day', 'week_of_year'])
```

### assign_season
- Maps months to seasons:
  - Winter (Dec-Feb), Spring (Mar-May), Summer (Jun-Aug), Fall (Sep-Nov).
 
```python
def assign_season(self, month):
        return ['Winter', 'Spring', 'Summer', 'Fall'][(month % 12) // 3]
```

### flag_holiday_period

- Determines if a date falls within 35 days of a holiday.

```python
def flag_holiday_period(self, row):
        holidays = self.country_holidays.get(row['country'], [])
        holiday_dates = [pd.Timestamp(year=row['year'], month=m, day=d) for m, d in holidays]
        return any(0 < (hd - row['date']).days <= 35 for hd in holiday_dates)
```

### calculate_gdp_change

- Computes the year-over-year GDP percentage change.

```python
    def calculate_gdp_change(self, row):
        gdp_data = self.gdp_dict.get(row['country'], {})
        return ((gdp_data.get(row['year'], 0) - gdp_data.get(row['year'] - 1, 0)) / gdp_data.get(row['year'] - 1, 1)) * 100
```

## Applying the Functions
- Creates an instance of DateFeatureEngineer for both training and test datasets.
- Calls extract_date_features() to generate new features.

```python
train = DateFeatureEngineer(train).extract_date_features()
test = DateFeatureEngineer(test).extract_date_features()
```
