# 🏆📈 Machine Learning Challenges: Forecasting Multiyear Sales with Advanced Date Feature Engineering and Ensemble Modeling
**This documentation reviews our innovative approach to tackle the Kaggle Sticker Sales Forecasting Challenge.**

*https://www.kaggle.com/competitions/playground-series-s5e1*

*Authors: Maksymilian Wiciński, Adam Tokarski*
### ⭐ Introduction

This document details the approach used to address the Kaggle Sticker Sales Forecasting Challenge, which involves predicting multiple years of sales for Kaggle-branded stickers across various fictitious stores and real-world countries. The synthetic dataset captures several real-world dynamics, including weekend and holiday effects, seasonality, and macroeconomic influences, making it an excellent testbed for advanced forecasting techniques.

🎯 **Objective:**  
Develop an end-to-end machine learning pipeline that includes comprehensive data preprocessing, advanced date feature engineering, and ensemble modeling to forecast future sticker sales.

🔍 **Data Engineering & Feature Extraction:**  
- **Date Features:** Extracted detailed date components (year, month, day, weekday, etc.) and created binary indicators for weekends and holidays.
- **Cyclical Transformations:** Applied sine and cosine transformations to capture cyclical patterns in days, months, and the overall year.
- **Economic Context:** Integrated external GDP data to include macroeconomic trends, calculating year-over-year changes to enhance feature relevance.

🤖 **Modeling Approach:**  
- **Segmented Modeling:** Divided the data by products, stores, and countries to train specific LightGBM models tailored to each segment.
- **Target Encoding:** Implemented target encoding for categorical variables to improve model performance and handle heterogeneous data.
- **Evaluation Metrics:** Employed early stopping and Mean Absolute Percentage Error (MAPE) as key evaluation metrics to monitor model performance.

📊 **Visualization & Communication:**  
- Created visualizations to explore temporal trends, feature distributions, and model evaluation results.
- Maintained detailed documentation of each step to ensure reproducibility and clarity in the workflow.

🛠️ **Tools & Technologies:**  
- **Programming Environment:** Python, using libraries such as pandas, NumPy, scikit-learn, and LightGBM.
- **Visualization:** Matplotlib and Seaborn for data exploration and result presentation.
- **Workflow Management:** Structured code into modular components to facilitate scalability and maintenance.

This documentation serves as a comprehensive record of the technical process, highlighting practical skills in data processing, feature engineering, and model development within a competitive machine learning context.

# 🎉 Final Version

<table style="border-collapse: collapse; border: 1px solid black; border-radius: 15px; background-color: #cce7ff; padding: 5px;">
  <tr>
    <td style="text-align: center; font-family: Arial, sans-serif; font-size: 50;">
      <img src="https://github.com/user-attachments/assets/7e141746-d572-4e52-96cf-31a1f01f3a97" alt="Icon" style="width: 20px; vertical-align: down; margin-right: 10px;">
      <b>Our Final Kaggle MAPE Score: 0.10609</b>
    </td>
  </tr>
</table>

<table style="border-collapse: collapse; border: 1px solid black; border-radius: 15px; background-color: #cce7ff; padding: 5px;">
  <tr>
    <td style="text-align: center; font-family: Arial, sans-serif; font-size: 50;">
      <img src="https://github.com/user-attachments/assets/7e141746-d572-4e52-96cf-31a1f01f3a97" alt="Icon" style="width: 20px; vertical-align: down; margin-right: 10px;">
      <b>1<sup>st</sup> Place Kaggle MAPE Score: 0.04265</b>
    </td>
  </tr>
</table>

## 🏗️ Class Definition & Initialization (__init__)
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

## 🔧 Function Definitions
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

## 🚀 Applying the Functions
- Creates an instance of DateFeatureEngineer for both training and test datasets.
- Calls extract_date_features() to generate new features.

```python
train = DateFeatureEngineer(train).extract_date_features()
test = DateFeatureEngineer(test).extract_date_features()
```

## 🔄 Final Column Adjustments

- Extracts `test_ids` before dropping the `id` column to retain them for submission.
- Retrieves unique values of categorical features (`products`, `stores`, `countries`) for reference.
- Removes the `id` column from both datasets since it does not contribute to predictions.
- Splits `train` into:
  - `X_train`: Feature set (all columns except `num_sold`).
  - `y_train`: Target variable (`num_sold`).
- Prepares `X_test` as the cleaned test dataset, ready for model inference.

```python
test_ids = test['id'].values

products = train['product'].unique().tolist()
stores = train['store'].unique().tolist()
countries = train['country'].unique().tolist()

train = train.drop(columns=['id'])
test = test.drop(columns=['id'])

X_train = train.drop(columns=['num_sold'])
y_train = train['num_sold']
X_test = test
```

## 🤖 Class: ProductModelTrainer

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

        self.predictions = pd.DataFrame()
        self.predictions_product = pd.DataFrame()
        self.predictions_store = pd.DataFrame()
        self.predictions_country = pd.DataFrame()

        self.categorical_cols_list = self.X_train_raw.select_dtypes(include=['object']).columns.intersection(
            self.X_test_raw.columns
        ).tolist()

        self.target_encoder = TargetEncoder(cols=self.categorical_cols_list)

        self.target_encoders_products = {}
        self.target_encoders_stores = {}
        self.target_encoders_countries = {}

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

## ✂️ Data Splitting Methods
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
## 📈 Training & Evaluation
### train_and_evaluate()
- Selects the model based on the `model_type` (product/store/country).
- Fits the model to training data with early stopping.
- Computes and prints **Mean Absolute Percentage Error (MAPE)**.
- Encodes test data using the stored `TargetEncoder`.
- Makes predictions on the test data.
- Saves predictions in the respective DataFrame.

```python
def train_and_evaluate(self, name, X_train_split, X_val_split, y_train_split, y_val_split, model_type):
        
        if model_type == 'product':
            model = self.product_models.get(name)
        elif model_type == 'store':
            model = self.store_models.get(name)
        elif model_type == 'country':
            model = self.country_models.get(name)
        else:
            raise ValueError("Unsupported model type: " + model_type)
        
        try:
            model.fit(X_train_split, y_train_split, eval_set=[(X_val_split, y_val_split)], early_stopping_rounds=10, verbose=False)
        except TypeError:
            model.fit(X_train_split, y_train_split)

        if model_type == 'product':
            self.product_models[name] = model
        elif model_type == 'store':
            self.store_models[name] = model
        elif model_type == 'country':
            self.country_models[name] = model

        y_val_pred = model.predict(X_val_split)
        y_val_pred = np.expm1(y_val_pred)
        y_val_actual = np.expm1(y_val_split)
        mape = mean_absolute_percentage_error(y_val_actual, y_val_pred)
        print(f"MAPE for {name}: {mape:.4f}")

        if model_type == 'product':
            test_mask = self.X_test_raw['product'] == name
        elif model_type == 'store':
            test_mask = self.X_test_raw['store'] == name
        elif model_type == 'country':
            test_mask = self.X_test_raw['country'] == name
        else:
            raise ValueError("Unsupported model type: " + model_type)

        X_test_subset = self.X_test_raw.loc[test_mask].copy()
        # Encode test data using the appropriate encoder.
        if model_type == 'product':
            X_test_encoded = self.target_encoders_products[name].transform(X_test_subset)
        elif model_type == 'store':
            X_test_encoded = self.target_encoders_stores[name].transform(X_test_subset)
        elif model_type == 'country':
            X_test_encoded = self.target_encoders_countries[name].transform(X_test_subset)
        else:
            raise ValueError("Unsupported model type: " + model_type)

        test_predictions = model.predict(X_test_encoded)
        test_predictions = np.expm1(test_predictions)
        test_ids = self.test_ids[test_mask.values]
        pred_df = pd.DataFrame({'id': test_ids, 'num_sold': test_predictions})

        if model_type == 'product':
            self.predictions_product = pd.concat([self.predictions_product, pred_df], ignore_index=True)
        elif model_type == 'store':
            self.predictions_store = pd.concat([self.predictions_store, pred_df], ignore_index=True)
        elif model_type == 'country':
            self.predictions_country = pd.concat([self.predictions_country, pred_df], ignore_index=True)
```
## 💾 Merging & Saving Predictions
### merge_predictions_and_save()
- Merges product, store, and country predictions based on `id`.
- Computes the median prediction across different models.
- Rounds `num_sold` values and saves the final CSV submission.

```python
def merge_predictions_and_save(self, submission_path):
        # Ensure all prediction dataframes have been populated.
        if self.predictions_product.empty or self.predictions_store.empty or self.predictions_country.empty:
            raise ValueError("One or more prediction dataframes are empty.")

        pred_product = self.predictions_product.rename(columns={'num_sold': 'num_sold_product'})
        pred_store = self.predictions_store.rename(columns={'num_sold': 'num_sold_store'})
        pred_country = self.predictions_country.rename(columns={'num_sold': 'num_sold_country'})

        merged = pred_product.merge(pred_store, on='id', how='outer')
        merged = merged.merge(pred_country, on='id', how='outer')

        merged['num_sold'] = merged[['num_sold_product', 'num_sold_store', 'num_sold_country']].median(axis=1)
        self.predictions = merged[['id', 'num_sold']].sort_values(by='id').reset_index(drop=True)
        self.predictions['num_sold'] = self.predictions['num_sold'].round().astype(int)
        self.predictions.to_csv(submission_path, index=False)
        print('Submission saved to', submission_path)
```

## 🔄 Main Processing Pipeline
### process()
- Splits the training data into subsets for:
  - Products
  - Stores
  - Countries
- Trains and evaluates models for each category.
- Calls `merge_predictions_and_save()` to generate the final submission.

```python
def process(self):
        # Split the training data for each category.
        subsets_product = self.split_data_by_product()
        subsets_store = self.split_data_by_store()
        subsets_country = self.split_data_by_country()

        for product in self.product_models.keys():
            if product in subsets_product:
                X_subset, y_subset = subsets_product[product]
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_subset, y_subset, test_size=0.2, random_state=42
                )
                self.train_and_evaluate(product, X_train_split, X_val_split, y_train_split, y_val_split,
                                        model_type='product')
            else:
                print(f"Skipping {product} because no training data is available.")

        for store in self.store_models.keys():
            if store in subsets_store:
                X_subset, y_subset = subsets_store[store]
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_subset, y_subset, test_size=0.2, random_state=42
                )
                self.train_and_evaluate(store, X_train_split, X_val_split, y_train_split, y_val_split,
                                        model_type='store')
            else:
                print(f"Skipping {store} because no training data is available.")

        for country in self.country_models.keys():
            if country in subsets_country:
                X_subset, y_subset = subsets_country[country]
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_subset, y_subset, test_size=0.2, random_state=42
                )
                self.train_and_evaluate(country, X_train_split, X_val_split, y_train_split, y_val_split,
                                        model_type='country')
            else:
                print(f"Skipping {country} because no training data is available.")
```
## ▶️ Executing the functions
- Creates an instance of `ProductModelTrainer` using neccessery parameters: `X_train`, `y_train`, `X_test`, `products`, `test_ids`, `stores`, `countries`.
- Calls `process()` to activate all the functions: `plit_data_by_product()`, `split_data_by_store()`, `split_data_by_country()`, `train_and_evaluate()`.
- Calls the final method `merge_predictions_and_save()` to properly save predictions to `submission.csv`.

```python
trainer = ProductModelTrainer(X_train, y_train, X_test, products, test_ids, stores, countries)
trainer.process()
trainer.merge_predictions_and_save("submission.csv")
```

# 📊 *Data Overview*
## Kaggle Data Source

This synthetic dataset contains multi-year sticker sales across different countries and stores, reflecting real-world patterns like weekends, holidays, and seasonality.

**Files**
- train.csv: Historical sales for each date, country, store, and item
- test.csv: Future data requiring predicted sales (num_sold)
- sample_submission.csv: Example of the required submission format

## **Data visualization and analysis using matplotlib**

This chart shows a single time series of total sales spanning from roughly 2009 to 2017, with values ranging between about 50,000 and 110,000. There are multiple pronounced peaks that reach near 90,000 to 110,000, as well as troughs dipping close to 60,000. Overall, the data exhibits a recurring, wavelike pattern, suggesting some form of seasonality or cyclical behavior over the years.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8fb13184-8807-4b44-88fc-c80756eec653">
</p>

This chart shows time series data for total sales (left y-axis) and GDP (right y-axis) in six countries—Canada, Italy, Norway, Finland, Kenya, and Singapore—from 2009 to 2017. Norway has the highest sales and GDP overall, while Kenya remains lowest in both measures. The sales curves fluctuate frequently, whereas the GDP lines change in larger “steps,” reflecting their less frequent reporting intervals.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9112aeb3-2a8d-4274-8518-39bfca230e85">
</p>

This pie chart illustrates the proportion of missing vs. non‐missing values in the num_sold column. A large blue slice (roughly 96.1%) represents the non‐missing values, while the smaller red slice (about 3.9%) accounts for the missing data, which we decided to delete.

<p align="center">
  <img src="https://github.com/user-attachments/assets/de560c90-bca0-4b88-9faf-d7516221cef0">
</p>
