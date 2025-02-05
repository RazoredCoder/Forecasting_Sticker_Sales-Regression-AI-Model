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
- Extracts date components: `year`, `month`, `day`, `day of the week`, `week of the year`, and `day of the year`.
- Creates binary indicators:
  - `is_weekend`: Checks if the day is Saturday (5) or Sunday (6).
  - `is_holiday`: Checks if the date is a holiday for the respective country.
- Adds season classification based on month.
- Defines `holiday_period` to mark a broader holiday season (±35 days from major holidays).
- Applies cyclical encoding using sine and cosine transformations:
  - `day_of_week_sin` & `day_of_week_cos`
  - `month_sin` & `month_cos`
  - `day_of_year_sin` & `day_of_year_cos`
- Fetches GDP values and computes the GDP percentage change from the previous year.
- Drops unnecessary columns: `date`, `month`, `day`, and `week_of_year`.

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
  - `Winter` (Dec-Feb), `Spring` (Mar-May), `Summer` (Jun-Aug), `Fall` (Sep-Nov).
 
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

## Final Column Adjustments

- Extracts `test_ids` before dropping the `id` column to retain them for submission.
- Retrieves unique values of categorical features (`products`, `stores`, `countries`) for reference.
- Removes the `id` column from both datasets since it does not contribute to predictions.
- Splits `train` into:
  - `X_train`: Feature set (all columns except `num_sold`).
  - `y_train`: Target variable (`num_sold`).
- Prepares `X_test` as the cleaned test dataset, ready for model inference.

```python
# Extract test IDs before dropping
test_ids = test['id'].values

# Get unique values of categorical features
products = train['product'].unique().tolist()
stores = train['store'].unique().tolist()
countries = train['country'].unique().tolist()

# Drop unnecessary columns
train = train.drop(columns=['id'])
test = test.drop(columns=['id'])

# Split training data into features (X) and target (y)
X_train = train.drop(columns=['num_sold'])
y_train = train['num_sold']
X_test = test
```

## Class: ProductModelTrainer

### Initialization (__init__ method)
- Stores raw training (`X_train`, `y_train`) and test (`X_test`) data.
- Applies log transformation to `y_train`.
- Stores lists of `products`, `test_ids`, `stores`, and `countries`.
- Initializes empty DataFrames for:
  - Overall predictions
  - Product-based predictions
  - Store-based predictions
  - Country-based predictions
- Identifies categorical columns present in both train and test data.
- Initializes a global `TargetEncoder` and dictionaries for storing encoders for products, stores, and countries.
- Defines parameters optimized with Optuna (`LGBMRegressor`) for:
  - Products (5 different models)
  - Stores (3 models)
  - Countries (6 models)

```python
class ProductModelTrainer:
    def __init__(self, X_train, y_train, X_test, products, test_ids, stores, countries):
        self.X_train_raw = X_train
        # Apply log transformation as in the original code.
        self.y_train = np.log1p(y_train + 1)
        self.X_test_raw = X_test
        self.products = products
        self.test_ids = test_ids
        self.stores = stores
        self.countries = countries

        # Initialize empty dataframes to hold predictions for each group.
        self.predictions = pd.DataFrame()
        self.predictions_product = pd.DataFrame()
        self.predictions_store = pd.DataFrame()
        self.predictions_country = pd.DataFrame()

        # Identify categorical columns present in both train and test.
        self.categorical_cols_list = self.X_train_raw.select_dtypes(include=['object']).columns.intersection(
            self.X_test_raw.columns
        ).tolist()

        # Global TargetEncoder (if needed)
        self.target_encoder = TargetEncoder(cols=self.categorical_cols_list)

        # Dictionaries to store TargetEncoders per product, store, or country.
        self.target_encoders_products = {}
        self.target_encoders_stores = {}
        self.target_encoders_countries = {}

        # Predefined models for products
        self.product_models = {
            "Kaggle": LGBMRegressor(
                random_state=0,
                n_estimators=365,
                max_depth=6,
                learning_rate=0.09951902014219748,
                num_leaves=93,
                subsample=0.8575678510026675,
                colsample_bytree=0.7836591837611223,
                reg_alpha=0.43350937968877423,
                reg_lambda=0.19887648661222768,
                min_child_weight=9
        # ETC. FULL VERSION IN THE REPOSITORY FILES
        self.store_models = {
        # ETC. FULL VERSION IN THE REPOSITORY FILES
        self.country_models = {
        # ETC. FULL VERSION IN THE REPOSITORY FILES
```

## Data Splitting Methods
### split_data_by_product()
- Checks if the `product` column exists in X_train.
- Creates subsets of data per product and applies Target Encoding.
- Saves the fitted encoder for later use.
```python
def split_data_by_product(self):
        subsets = {}
        if 'product' not in self.X_train_raw.columns:
            raise KeyError("'product' column is missing in X_train.")
        for product in self.products:
            product_mask = self.X_train_raw['product'] == product
            if product_mask.sum() == 0:
                print(f"No data found for product: {product}")
                continue
            X_subset = self.X_train_raw.loc[product_mask].copy()
            y_subset = self.y_train.loc[product_mask]
            encoder = TargetEncoder(cols=self.categorical_cols_list)
            X_subset_encoded = encoder.fit_transform(X_subset, y_subset)
            self.target_encoders_products[product] = encoder  # Save encoder for test-time.
            subsets[product] = (X_subset_encoded, y_subset)
        return subsets
```

### split_data_by_store()
- Checks for `store` column.
- Creates subsets of data per store and applies Target Encoding.
- Stores the fitted encoder.
```python
def split_data_by_store(self):
        subsets = {}
        if 'store' not in self.X_train_raw.columns:
            raise KeyError("'store' column is missing in X_train.")
        for store in self.stores:
            store_mask = self.X_train_raw['store'] == store
            if store_mask.sum() == 0:
                print(f"No data found for store: {store}")
                continue
            X_subset = self.X_train_raw.loc[store_mask].copy()
            y_subset = self.y_train.loc[store_mask]
            encoder = TargetEncoder(cols=self.categorical_cols_list)
            X_subset_encoded = encoder.fit_transform(X_subset, y_subset)
            self.target_encoders_stores[store] = encoder
            subsets[store] = (X_subset_encoded, y_subset)
        return subsets
```

### split_data_by_country()
- Checks for `country` column.
- Creates subsets per country and applies Target Encoding.
- Saves the encoder for later use.
```python
def split_data_by_country(self):
        subsets = {}
        if 'country' not in self.X_train_raw.columns:
            raise KeyError("'country' column is missing in X_train.")
        for country in self.countries:
            country_mask = self.X_train_raw['country'] == country
            if country_mask.sum() == 0:
                print(f"No data found for country: {country}")
                continue
            X_subset = self.X_train_raw.loc[country_mask].copy()
            y_subset = self.y_train.loc[country_mask]
            encoder = TargetEncoder(cols=self.categorical_cols_list)
            X_subset_encoded = encoder.fit_transform(X_subset, y_subset)
            self.target_encoders_countries[country] = encoder  # Save encoder for later.
            subsets[country] = (X_subset_encoded, y_subset)
        return subsets
```
